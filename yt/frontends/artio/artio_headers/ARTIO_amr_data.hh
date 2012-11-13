/*
		This file is part of libARTIO++ 
			a C++ library to access snapshot files 
			generated by the simulation code ARTIO by R. Teyssier
		
    Copyright (C) 2008-09  Oliver Hahn, ojha@gmx.de

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ARTIO_AMR_DATA_HH
#define __ARTIO_AMR_DATA_HH

#include <fstream>
#include <iostream>
#include <iomanip>
#include <vector>
#include <map>
#include <cmath>
#include <iterator>

#include "FortranUnformatted_IO.hh"
#include "ARTIO_info.hh"
#include "ARTIO_amr_data.hh"

#ifndef FIX
#define FIX(x)	((int)((x)+0.5))
#endif

#define ACC_NL(cpu,lvl)   ((cpu)+m_header.ncpu*(lvl))
#define ACC_NLIT(cpu,lvl) ((cpu)+m_plevel->m_header.ncpu*(lvl))
#define ACC_NB(cpu,lvl)   ((cpu)+m_header.nboundary*(lvl))

#define LENGTH_POINTERLISTS 4096
#define ENDPOINT ((unsigned)(-1))

namespace ARTIO{
namespace AMR{
	
/**************************************************************************************\
 *** auxiliary datatypes **************************************************************
\**************************************************************************************/
	
template< typename real_t>
struct vec{
	real_t x,y,z;
		
	vec( real_t x_, real_t y_, real_t z_ )
	: x(x_),y(y_),z(z_)
	{ }
		
	vec( const vec& v )
	: x(v.x),y(v.y),z(v.z)
	{ }
		
	vec( void )
	: x(0.0), y(0.0), z(0.0)
	{ }
};

/**************************************************************************************\
 *** AMR cell base types **************************************************************
\**************************************************************************************/

template <typename id_t=unsigned, typename real_t=float>
class cell_locally_essential{
public:
	id_t m_neighbour[6];
	id_t m_father;
	id_t m_son[8];
	real_t m_xg[3];
	id_t m_cpu;

	char m_pos;
	
	cell_locally_essential(){}
		
	bool is_refined( int ison ) const
	{	return ((int)m_son[ison]!=-1); }	
};

template <typename id_t=unsigned, typename real_t=float>
class cell_simple{
public:
	id_t m_son[1];
	real_t m_xg[3];
	id_t m_cpu;

	char m_pos;
	
	cell_simple(){}
		
	bool is_refined( int ison=0 ) const
	{	return ((int)m_son[0]!=-1); }	
};


//.... some type traits that are used to distinguish what data needs to be read ....//
template<class X> struct is_locally_essential
{ enum { check=false }; };

template<> struct is_locally_essential<cell_locally_essential<> > 
{ enum { check=true }; };

template<> struct is_locally_essential<cell_simple<> > 
{ enum { check=false }; };


/**************************************************************************************\
 *** AMR level class definition, subsumes a collection of AMR cells *******************
\**************************************************************************************/

//! AMR level implementation
template< typename Cell_ >
class level{
public:
	unsigned m_ilevel;
	std::vector< Cell_ > m_level_cells;
	
	double m_xc[8];			//!< relative x-offsets of the 8 children for current refinement level
	double m_yc[8];			//!< relative y-offsets of the 8 children for current refinement level
	double m_zc[8];			//!< relative z-offsets of the 8 children for current refinement level
	
	typedef typename std::vector< Cell_ >::iterator iterator;
	typedef typename std::vector< Cell_ >::const_iterator const_iterator;

	double 		m_dx;
	unsigned 	m_nx;

	level( unsigned ilevel )
	: m_ilevel( ilevel )
	{
		m_dx = pow(0.5,ilevel+1);
		m_nx = (unsigned)(1.0/m_dx);
		
		for( unsigned k=1; k<=8; k++ )
		{
			//... initialize positions of son cell centres
			//... relative to parent cell centre
			unsigned 
				iz=(k-1)/4,
				iy=(k-1-4*iz)/2,
				ix=(k-1-2*iy-4*iz);
			m_xc[k-1]=((double)ix-0.5f)*m_dx;
			m_yc[k-1]=((double)iy-0.5f)*m_dx;
			m_zc[k-1]=((double)iz-0.5f)*m_dx;
		}
	}
	
	void register_cell( const Cell_ & cell )
	{ m_level_cells.push_back( cell ); }
	
	const_iterator begin( void ) const{ return m_level_cells.begin(); }
	iterator begin( void ){ return m_level_cells.begin(); }
	
	const_iterator end( void ) const{ return m_level_cells.end(); }
	iterator end( void ){ return m_level_cells.end(); }
	
	Cell_& operator[]( unsigned i )
	{ return m_level_cells[i]; }
	
	unsigned size( void ) { return m_level_cells.size(); }
};


/**************************************************************************************\
 *** constants ************************************************************************
\**************************************************************************************/

//! neighbour cell access pattern
const static int nbor_cell_map[6][8] = 
{ 
	{1,0,3,2,5,4,7,6},
	{1,0,3,2,5,4,7,6},
	{2,3,0,1,6,7,4,5},
	{2,3,0,1,6,7,4,5},
	{4,5,6,7,0,1,2,3},
	{4,5,6,7,0,1,2,3} 
};


/**************************************************************************************\
 *** AMR tree class implements the hierarchy of AMR levels with links *****************
\**************************************************************************************/

/*!
 * @class tree
 * @brief encapsulates the hierarchical AMR structure data from a ARTIO simulation snapshot
 *
 * This class provides low-level read access to ARTIO amr_XXXXX.out files. 
 * Data from a given list of computational domains can be read and is
 * stored in internal datastructures.
 * Access to hydrodynamical variables stored in the cells is provided 
 * through member functions of class ARTIO_hydro_level and iterators 
 * provided by this class
 * @sa ARTIO_hydro_level
 */
template< class Cell_, class Level_ >
class tree
{

public:
	
	//! header amr meta-data structure, for details see also ARTIO source code (file amr/init_amr.f90)
	struct header{ 
		std::vector< int > nx;			//!< base mesh resolution [3-vector]
		std::vector< int > nout;		//!< 3 element vector: [noutput2, iout2, ifout2]
		std::vector< int > nsteps;		//!< 2 element vector: [nstep, nstep_coarse]
		int ncpu;						//!< number of CPUs (=computational domains) in the simulation
		int ndim;						//!< number of spatial dimensions
		int nlevelmax;					//!< maximum refinement level allowed
		int ngridmax;					//!< maxmium number of grid cells stored per CPU
		int nboundary;					//!< number of boundary cells (=ghost cells)
		int ngrid_current;				//!< currently active grid cells
		double boxlen;					//!< length of the simulation box
		std::vector<double> tout;		//!< output times (1..noutput)
		std::vector<double> aout;		//!< output times given as expansion factors (1..noutput)
		std::vector<double> dtold;		//!< old time steps (1..nlevelmax)
		std::vector<double> dtnew;		//!< next time steps (1..nlevelmax)
		std::vector<double> stat;		//!< some diagnostic snapshot meta data: [const, mass_tot0, rho_tot]
		std::vector<double> cosm;		//!< cosmological meta data: [omega_m, omega_l, omega_k, omega_b, h0, aexp_ini, boxlen_ini)
		std::vector<double> timing;		//!< timing information: [aexp, hexp, aexp_old, epot_tot_int, epot_tot_old]
		double t;						//!< time stamp of snapshot
		double mass_sph;				//!< mass threshold used to flag for cell refinement
	};
	
	std::vector< Level_ > m_AMR_levels;	//! STL vector holding the AMR level hierarchy
	
	std::vector<unsigned> 
		m_headl,						//!< head indices, point to first active cell
		m_numbl, 						//!< number of active cells
		m_taill;						//!< tail indices, point to last active cell

	int m_cpu;							//! index of computational domain being accessed
	int m_minlevel;						//! lowest refinement level to be loaded
	int m_maxlevel;						//! highest refinement level to be loaded
	std::string m_fname;				//! the snapshot filename amr_XXXXX.out
	unsigned m_ncoarse;					//! number of coarse grids
	struct header m_header;				//! the header meta data
	
	
protected:
	
  //! prototypical grid iterator
  /*! iterates through cells on one level, provides access to neighbours, parents, children
   */
	template< typename TreePointer_, typename Index_=unsigned >
	class proto_iterator{
		public:
		
			friend class tree;
			
			typedef Index_ value_type;
			typedef Index_& reference;
			typedef Index_* pointer;
			
		protected:
		
			Index_
				m_ilevel,		//!< refinement level on which we iteratre
				m_icpu;			//!< domain on which we iterate
				
			typedef typename std::vector< Cell_ >::const_iterator Iterator_;
			Iterator_	 m_cell_it; //!< basis iterator that steps through cells on one level
			TreePointer_ m_ptree; //!< pointer to associated tree object
			
			
			//! low_level construtor that should only be used from within AMR_tree
			proto_iterator( unsigned level_, unsigned cpu_, Iterator_ it_, TreePointer_ ptree_ )
			: m_ilevel(level_), m_icpu(cpu_), m_cell_it(it_), m_ptree( ptree_ )
			{ }
			
		public:
		
			//! this is either the copy-constructor or a constructor for implicit type conversion
			template< typename X >
			proto_iterator( proto_iterator<X> &it )
			: m_ilevel( it.m_ilevel ), m_icpu( it.m_icpu ), m_cell_it( it.m_cell_it ), m_ptree( it.m_ptree )
			{ }
	
			//! empty constructor, doesn't initialize anything
			proto_iterator()
			: m_ilevel(0), m_icpu(0), m_ptree(NULL)
			{ }
			
			//! test for equality between two proto_iterator instantiations
			template< typename X >
			inline bool operator==( const proto_iterator<X>& x ) const
			{ return m_cell_it==x.m_cell_it; }

			//! test for inequality between two proto_iterator instantiations	
			template< typename X >
			inline bool operator!=( const proto_iterator<X>& x ) const
			{ return m_cell_it!=x.m_cell_it; }

			//! iterate forward, prefix
			inline proto_iterator& operator++()
			{ ++m_cell_it; return *this; }
			
			//! iterate forward, postfix
			inline proto_iterator operator++(int)
			{ proto_iterator<TreePointer_> it(*this); operator++(); return it; }

            inline void next(void) { operator++(); }
			
			//! iterate backward, prefix
			inline proto_iterator& operator--()
			{ --m_cell_it; return *this; }
			
			//! iterate backward, postfix
			inline proto_iterator operator--(int)
			{ proto_iterator<TreePointer_> it(*this); operator--(); return it; }
			
			//! iterate several forward
			inline proto_iterator operator+=(int i)
			{ proto_iterator<TreePointer_> it(*this); m_cell_it+=i; return it; }
			
			//! iterate several backward
			inline proto_iterator operator-=(int i)
			{ proto_iterator<TreePointer_> it(*this); m_cell_it-=i; return it; }
			
			//! assign two proto_iterators, this will fail if no typecast between X and TreePoint_ exists
			template< typename X >
			inline proto_iterator& operator=(const proto_iterator<X>& x)
			{ m_cell_it = x.m_cell_it; m_ilevel = x.m_ilevel; m_icpu = x.m_icpu; m_ptree = x.m_ptree; return *this; }
			
			//! iterator dereferencing, returns an array index
			inline Cell_ operator*() const
			{ return *m_cell_it; }
			
            inline Index_ get_cell_father() const { return (*m_cell_it).m_father; }
            inline bool is_finest(int ison) { return !((*m_cell_it).is_refined(ison)); }
			
			//! move iterator to a child grid
			/*!
			 * @param  ind index of the child grid in the parent oct (0..7)
			 * @return iterator pointing to child grid if it exists, otherwise 'end' of currrent level
			 */
			//template< typename X >
			inline proto_iterator& to_child( int ind )
			{
				if( !(*m_cell_it).is_refined(ind) )
					 return (*this = m_ptree->end( m_ilevel ));
				++m_ilevel;
				m_cell_it = m_ptree->m_AMR_levels[m_ilevel].begin()+(*m_cell_it).m_son[ind];
				return *this;
			}
			
			
			//! get an iterator to a child grid
			/*!
			 * @param  ind index of the child grid in the parent oct (0..7)
			 * @return iterator pointing to child grid if it exists, otherwise 'end' of currrent level
			 */
			//template< typename X >
			inline proto_iterator get_child( int ind ) const
			{
				proto_iterator it(*this);
				it.to_child( ind );
				return it;
			}
			
			inline char get_child_from_pos( double x, double y, double z ) const
			{
				bool  bx,by,bz;
				bx = x > (*m_cell_it).m_xg[0];
				by = y > (*m_cell_it).m_xg[1];
				bz = z > (*m_cell_it).m_xg[2];
				
				//std::cerr << "(" << bx << ", " << by << ", " << bz << ")\n";
				
				return (char)bx+2*((char)by+2*(char)bz);
			}
			
			
			//! move iterator to the parent grid
			/*! 
			 * @return iterator pointing to the parent grid if it exists, 'end' of the current level otherwise
			 */
			
			inline proto_iterator& to_parent( void )
			{
				if( m_ilevel==0 )
					return (*this = m_ptree->end( m_ilevel ));
				--m_ilevel;
				m_cell_it = m_ptree->m_AMR_levels[m_ilevel].begin()+(*m_cell_it).m_father;
				return *this;
			}
			
			//! query whether a given child cell is refined
			inline bool is_refined( int i ) const
			{
				return (*m_cell_it).is_refined(i);
			}
			
			//! get an iterator to the parent grid
			/*! 
			 * @return iterator pointing to the parent grid if it exists, 'end' of the current level otherwise
			 */
			inline proto_iterator get_parent( void ) const
			{
				proto_iterator it(*this);
				it.to_parent();
				return it;
			}
			
			//! move iterator to spatial neighbour grid
			/*!
			 * @param ind index of neighbour (0..5)
			 * @return iterator pointing to neighbour grid if it exists, otherwise 'end' of currrent level
			 */
			inline proto_iterator& to_neighbour( int ind )
			{
				unsigned icell = nbor_cell_map[ind][(int)(*m_cell_it).m_pos];
				m_cell_it = m_ptree->m_AMR_levels[m_ilevel-1].begin()+(*m_cell_it).m_neighbour[ind];
				
				if( !(*m_cell_it).is_refined(icell) )
					return (*this = m_ptree->end(m_ilevel));
				
				m_cell_it = m_ptree->m_AMR_levels[m_ilevel].begin()+(*m_cell_it).m_son[icell];
				return *this;
			}
			
			//! get an iterator to spatial neighbour grid
			/*!
			 * @param ind index of neighbour (0..5)
			 * @return iterator pointing to neighbour grid if it exists, otherwise 'end' of currrent level
			 */
			inline proto_iterator& get_neighbour( int ind )
			{
				proto_iterator it(*this);
				it.to_neighbour(ind);
				return it;
			}
			
			inline Index_ get_level( void ) const
			{ return m_ilevel; }
			
			inline int get_domain( void ) const
			{ return (*m_cell_it).m_cpu; }
			
			inline int get_absolute_position( void ) const 
			{
				return (unsigned)(std::distance<Iterator_>(m_ptree->m_AMR_levels[m_ilevel].begin(),m_cell_it));
			}
			
			
	};
	
public:
	
	typedef proto_iterator<const tree*> const_iterator;
	typedef proto_iterator<tree*>       iterator;
	


protected:
	
	//! read header meta data from amr snapshot file
	void read_header( void );
	

	
	//! generate the amr_XXXXX.out filename for a given computational domain
	/*! @param icpu index of comutational domain (base 1)
	 */
	std::string gen_fname( int icpu );
	
	//! generate the amr_XXXXX.out filename from the path to the info_XXXXX.out file
	std::string rename_info2amr( const std::string& info );


	#define R_SQR(x) ((x)*(x))
	
	template< typename Real_ >
	inline bool ball_intersection( const vec<Real_>& xg, double dx2, const vec<Real_>& xc, Real_ r2 )
	{
		Real_ dmin = 0, bmin, bmax;
		
		//.. x ..//
		bmin = xg.x-dx2;
		bmax = xg.x+dx2;
		if( xc.x < bmin ) dmin += R_SQR(xc.x - bmin ); else
		if( xc.x > bmax ) dmin += R_SQR(xc.x - bmax );
			
		//.. y ..//
		bmin = xg.y-dx2;
		bmax = xg.y+dx2;
		if( xc.y < bmin ) dmin += R_SQR(xc.y - bmin ); else
		if( xc.y > bmax ) dmin += R_SQR(xc.y - bmax );
		
		//.. x ..//
		bmin = xg.z-dx2;
		bmax = xg.z+dx2;
		if( xc.z < bmin ) dmin += R_SQR(xc.z - bmin ); else
		if( xc.z > bmax ) dmin += R_SQR(xc.z - bmax );
		
		if( dmin <= r2 ) return true;
		return false;
	}
	
	template< typename Real_ >
	inline bool shell_intersection( const vec<Real_>& xg, double dx2, const vec<Real_>& xc, Real_ r1_2, Real_ r2_2 )
	{
		Real_ dmax = 0, dmin = 0, a, b, bmin, bmax;
		if( r1_2 > r2_2 ) std::swap(r1_2,r2_2);
			
		//.. x ..//
		bmin = xg.x-dx2;
		bmax = xg.x+dx2;
		a = R_SQR( xc.x - bmin );
		b = R_SQR( xc.x - bmax );
		dmax += std::max( a, b );
		if( xc.x < bmin ) dmin += a; else
		if( xc.x > bmax ) dmin += b;
		
		//.. y ..//
		bmin = xg.y-dx2;
		bmax = xg.y+dx2;
		a = R_SQR( xc.y - bmin );
		b = R_SQR( xc.y - bmax );
		dmax += std::max( a, b );
		if( xc.y < bmin ) dmin += a; else
		if( xc.y > bmax ) dmin += b;
			
		//.. z ..//
		bmin = xg.z-dx2;
		bmax = xg.z+dx2;
		a = R_SQR( xc.z - bmin );
		b = R_SQR( xc.z - bmax );
		dmax += std::max( a, b );
		if( xc.z < bmin ) dmin += a; else
		if( xc.z > bmax ) dmin += b;
		
		
		if( dmin <= r2_2 && r1_2 <= dmax ) return true;
		return false;
	}
	
	template< typename Real_ >
	inline bool sphere_intersection( const vec<Real_>& xg, double dx2, const vec<Real_>& xc, Real_ r2 )
	{
		Real_ dmax = 0, dmin = 0, a, b, bmin, bmax;
		
		//.. x ..//
		bmin = xg.x-dx2;
		bmax = xg.x+dx2;
		a = R_SQR( xc.x - bmin );
		b = R_SQR( xc.x - bmax );
		dmax += std::max( a, b );
		if( xc.x < bmin ) dmin += a; else
		if( xc.x > bmax ) dmin += b;
		
		//.. y ..//
		bmin = xg.y-dx2;
		bmax = xg.y+dx2;
		a = R_SQR( xc.y - bmin );
		b = R_SQR( xc.y - bmax );
		dmax += std::max( a, b );
		if( xc.y < bmin ) dmin += a; else
		if( xc.y > bmax ) dmin += b;
			
		//.. z ..//
		bmin = xg.z-dx2;
		bmax = xg.z+dx2;
		a = R_SQR( xc.z - bmin );
		b = R_SQR( xc.z - bmax );
		dmax += std::max( a, b );
		if( xc.z < bmin ) dmin += a; else
		if( xc.z > bmax ) dmin += b;
		
		
		if( dmin <= r2 && r2 <= dmax ) return true;
		return false;
	}
	
	#undef R_SQR
	
public:
	
	//! low-level constructor - should not be called from outside because then you can really screw up things
	/*!
	 * @param snap the associated ARTIO::snapshot object
	 * @param cpu domain for which to read the AMR tree
	 * @param maxlevel maximum refinement level to consider
	 * @param minlevel minimum refinement level to consider (default=1)
	 */
	tree( ARTIO::snapshot& snap, int cpu, int maxlevel, int minlevel=1 )
	: m_cpu( cpu ), m_minlevel( minlevel ), m_maxlevel( maxlevel ), m_fname( rename_info2amr(snap.m_filename) )
	{ 
		read_header();

		if( cpu > m_header.ncpu || cpu <= 0)
			throw std::runtime_error("ARTIO_particle_data: expect to read from out of range CPU.");
		
	}
	
	//! perform the read operation of AMR data
	void read( void );
	
	//! end const_iterator for given refinement level
	const_iterator end( int ilevel ) const
	{ 
		if( ilevel <= m_maxlevel )
			return const_iterator( ilevel, m_cpu, m_AMR_levels.at(ilevel).end(), this );
		else
			return const_iterator( ilevel, m_cpu, m_AMR_levels.at(0).end(), this );
	}
	
	//! end iterator for given refinement level
	iterator end( int ilevel )
	{ 
		if( ilevel <= m_maxlevel )
			return iterator( ilevel, m_cpu, m_AMR_levels.at(ilevel).end(), this ); 
		else
			return iterator( ilevel, m_cpu, m_AMR_levels.at(0).end(), this ); 
	}
	
	//! begin const_iterator for given refinement level
	const_iterator begin( int ilevel ) const
	{	
		if( ilevel <= m_maxlevel )
			return const_iterator( ilevel, m_cpu, m_AMR_levels.at(ilevel).begin(), this ); 
		else
			return this->end(ilevel);
	}
	
	//! begin iterator for given refinement level
	iterator begin( int ilevel )
	{	
		if( ilevel <= m_maxlevel )
			return iterator( ilevel, m_cpu, m_AMR_levels.at(ilevel).begin(), this ); 
		else
			return this->end(ilevel);
	}
	
	
	//! return the center of a child cell associated with a grid iterator
	/*!
	 * @param it grid iterator
	 * @param ind sub-cell index (0..7)
	 * @return vec vector containing the coordinates
	 */
	template< typename Real_ >
	inline vec<Real_> cell_pos( const iterator& it, unsigned ind )
	{
		vec<Real_> pos;
		pos.x = (*it).m_xg[0]+m_AMR_levels[it.m_ilevel].m_xc[ind];
		pos.y = (*it).m_xg[1]+m_AMR_levels[it.m_ilevel].m_yc[ind];
		pos.z = (*it).m_xg[2]+m_AMR_levels[it.m_ilevel].m_zc[ind];
		return pos;
	}
	
	//! return the center of the grid associated with a grid iterator
	/*!
	 * @param it grid iterator
	 * @return vec vector containing the coordinates
	 */
	template< typename Real_ >
	inline vec<Real_> grid_pos( const iterator& it )
	{
		vec<Real_> pos;
		pos.x = (*it).m_xg[0];
		pos.y = (*it).m_xg[1];
		pos.z = (*it).m_xg[2];
		return pos;
	}
	
	template< typename Real_ >
	inline bool ball_intersects_grid( const iterator& it, const vec<Real_>& xc, Real_ r2 )
	{
		Real_ dx2 = 0.5/pow(2,it.get_level());
		vec<Real_> xg = grid_pos<Real_>(it);
		return ball_intersection( xg, dx2, xc, r2 );
	}
	
	template< typename Real_ >
	inline bool ball_intersects_cell( const iterator& it, char ind, const vec<Real_>& xc, Real_ r2 )
	{
		Real_ dx2 = 0.5/pow(2,it.get_level()+1);
		vec<Real_> xg = cell_pos<Real_>(it,ind);
		return ball_intersection( xg, dx2, xc, r2 );
	}
		
	template< typename Real_ >
	inline bool shell_intersects_grid( iterator& it, const vec<Real_>& xc, Real_ r1_2, Real_ r2_2 )
	{
		Real_ dx2 = 0.5/pow(2,it.get_level());
		vec<Real_> xg = grid_pos<Real_>(it);
		return shell_intersection( xg, dx2, xc, r1_2, r2_2 );
	}
	
	template< typename Real_ >
	inline bool shell_intersects_cell( iterator& it, char ind, const vec<Real_>& xc, Real_ r1_2, Real_ r2_2 )
	{
		Real_ dx2 = 0.5/pow(2,it.get_level()+1);
		vec<Real_> xg = cell_pos<Real_>(it,ind);
		return shell_intersection( xg, dx2, xc, r1_2, r2_2 );
	}
		
	template< typename Real_ >
	inline bool sphere_intersects_grid( const iterator& it, const vec<Real_>& xc, Real_ r2 )
	{
		Real_ dx2 = 0.5/pow(2,it.get_level());
		vec<Real_> xg = grid_pos<Real_>(it);
		return sphere_intersection( xg, dx2, xc, r2 );
	}
	
	template< typename Real_ >
	inline bool sphere_intersects_cell( const iterator& it, char ind, const vec<Real_>& xc, Real_ r2 )
	{
		Real_ dx2 = 0.5/pow(2,it.get_level()+1);
		vec<Real_> xg = cell_pos<Real_>(it,ind);
		return sphere_intersection( xg, dx2, xc, r2 );
	}
	
};

/**************************************************************************************\
\**************************************************************************************/

template< class Cell_, class Level_ >
void tree<Cell_,Level_>::read_header( void )
{
    FortranUnformatted ff( gen_fname(m_cpu) );
    std::cout << "snl: HI I AM HERE AND ALIVE!!!!!!";
	
	//-- read header data --//
	
	ff.read( m_header.ncpu );
	ff.read( m_header.ndim );
	ff.read<unsigned>( std::back_inserter(m_header.nx) );
	ff.read( m_header.nlevelmax );
	ff.read( m_header.ngridmax );
	ff.read( m_header.nboundary );
	ff.read( m_header.ngrid_current );
	ff.read( m_header.boxlen );
	
	ff.read<unsigned>( std::back_inserter(m_header.nout) );
	ff.read<double>( std::back_inserter(m_header.tout) );
	ff.read<double>( std::back_inserter(m_header.aout) );
	ff.read( m_header.t );
	ff.read<double>( std::back_inserter(m_header.dtold) );
	ff.read<double>( std::back_inserter(m_header.dtnew) );
	ff.read<unsigned>( std::back_inserter(m_header.nsteps) );
	ff.read<double>( std::back_inserter(m_header.stat) );
	ff.read<double>( std::back_inserter(m_header.cosm) );
	ff.read<double>( std::back_inserter(m_header.timing) );
	ff.read( m_header.mass_sph );
	
	m_ncoarse = m_header.nx[0]*m_header.nx[1]*m_header.nx[2];
}

/**************************************************************************************\
\**************************************************************************************/

template< class Cell_, class Level_ >
std::string tree<Cell_,Level_>::gen_fname( int icpu )
{
	std::string fname;
	char ext[32];
	fname = m_fname;
	fname.erase(fname.rfind('.')+1);
	sprintf(ext,"out%05d",icpu);
	fname.append(std::string(ext));
	return fname;
}

/**************************************************************************************\
\**************************************************************************************/

template< class Cell_, class Level_ >
std::string tree<Cell_,Level_>::rename_info2amr( const std::string& info )
{
	std::string amr;
	unsigned ii = info.rfind("info");
	amr = info.substr(0,ii)+"amr" + info.substr(ii+4, 6) + ".out00001";
	return amr;
}

/**************************************************************************************\
\**************************************************************************************/

template< class Cell_, class Level_ >
void tree<Cell_,Level_>::read( void )
{
	// indexing map used to associate ARTIO internal cell IDs with new IDs
	std::map<unsigned,unsigned> m_ind_grid_map;

	std::vector<int> cell_cpu;
	std::vector<unsigned> cell_level;
	std::vector<unsigned> itmp;
	
	
	typename std::vector< Level_ >::iterator amr_level_it;
	
	FortranUnformatted ff( gen_fname( m_cpu ) );
		
	//.. skip header entries ..//
	ff.skip_n_from_start( 19 ); 			//.. skip header 
		
	//+ headl + taill
	ff.read<int>( std::back_inserter(m_headl) );
	ff.read<int>( std::back_inserter(m_taill) );
	ff.read<int>( std::back_inserter(m_numbl) );
	
	//.. skip numbtot
	ff.skip_n( 1 ); 						
		
	std::vector<int> ngridbound;
	if( m_header.nboundary > 0 ){
		ff.skip_n( 2 ); 					//.. skip headb and tailb 
		ff.read<int>( std::back_inserter(ngridbound) ); 				//.. read numbb
	}
		
	ff.skip_n( 6 ); //..skip (1)free_mem+(2)ordering+(3)bound_key+
					//..     (4)coarse_son+(5)coarse_flag1+(6)coarse_cpu_map
	
	
	if( /*m_minlevel < 1 ||*/ m_maxlevel > m_header.nlevelmax || m_minlevel > m_maxlevel )
		throw std::runtime_error("ARTIO_amr_level::read_level : requested level is invalid.");
		

	m_ind_grid_map.insert( std::pair<unsigned,unsigned>(0,ENDPOINT) );
	
	FortranUnformatted::streampos spos = ff.tellg();
	
	m_minlevel = 0;
	
	//... create indexing map ...//
	for( int ilvl = 0; ilvl<=std::min(m_maxlevel+1, m_header.nlevelmax); ++ilvl ){
		unsigned gridoff = 0;
		for( int icpu=0; icpu<m_header.ncpu+m_header.nboundary; ++icpu ){
			if( icpu < m_header.ncpu && m_numbl[ACC_NL(icpu,ilvl)] == 0 )
				continue;
			else if( icpu >= m_header.ncpu && ngridbound[ACC_NB(icpu-m_header.ncpu,ilvl)] == 0 )
				continue;
				
			if( ilvl >= m_minlevel ){
				std::vector<int> ind_grid;
				ff.read<int>( std::back_inserter(ind_grid) );
				for( unsigned i=0; i<ind_grid.size(); ++i ){
					m_ind_grid_map.insert( std::pair<unsigned,unsigned>(ind_grid[i],gridoff++) );
				}
				ind_grid.clear();
			}else
				ff.skip();
				
			ff.skip_n( 3+3+6+8+8+8 );
		}
		if( ff.eof() ){
			//std::cerr << "eof reached in fortran read operation\n";
			m_maxlevel = ilvl;//+1;
			break;
		}
	}
	
	ff.seekg( spos );
	
	m_AMR_levels.clear();
	
	 
	//... loop over levels ...//
	for( int ilvl = 0; ilvl<=m_maxlevel; ++ilvl ){
		m_AMR_levels.push_back( Level_(ilvl) );
		Level_ &currlvl = m_AMR_levels.back();
			
		for( int icpu=0; icpu<m_header.ncpu+m_header.nboundary; ++icpu ){
			if( icpu < m_header.ncpu && m_numbl[ACC_NL(icpu,ilvl)] == 0 )
				continue;
			else if( icpu >= m_header.ncpu && ngridbound[ACC_NB(icpu-m_header.ncpu,ilvl)] == 0 )
				continue;
			
			if( ilvl >= m_minlevel ){
				unsigned gridoff = currlvl.size();
			
				std::vector<int> ind_grid;
				ff.read<int>( std::back_inserter(ind_grid) );
				for( unsigned i=0; i<ind_grid.size(); ++i ){
					currlvl.register_cell( Cell_() );
					//.. also set owning cpu in this loop...
					currlvl[ i+gridoff ].m_cpu = icpu+1;
				}
			
				//... pointers to next and previous octs ..//
				ff.skip();
				ff.skip();
			
				//... oct x-coordinates ..//
				std::vector<float> ftmp;
				ff.read<double>( std::back_inserter(ftmp) );
				for( unsigned j=0; j<ftmp.size(); ++j )
					currlvl[ j+gridoff ].m_xg[0] = ftmp[j];
				ftmp.clear();
			
				//... oct y-coordinates ..//
				ff.read<double>( std::back_inserter(ftmp) );
				for( unsigned j=0; j<ftmp.size(); ++j )
					currlvl[ j+gridoff ].m_xg[1] = ftmp[j];
				ftmp.clear();
			
				//... oct y-coordinates ..//
				ff.read<double>( std::back_inserter(ftmp) );
				for( unsigned j=0; j<ftmp.size(); ++j )
					currlvl[ j+gridoff ].m_xg[2] = ftmp[j];
				ftmp.clear();
			
			
				//... father indices
				if( is_locally_essential<Cell_>::check ){
					ff.read<int>( std::back_inserter(itmp) ); 
					for( unsigned j=0; j<itmp.size(); ++j ){
						currlvl[ j+gridoff ].m_pos    = (itmp[j]-m_ncoarse-1)/m_header.ngridmax;
						currlvl[ j+gridoff ].m_father = m_ind_grid_map[ (itmp[j]-m_ncoarse)%m_header.ngridmax ];
					}
					itmp.clear();
				}else
					ff.skip();
				
				
				//... neighbour grids indices
				if( is_locally_essential<Cell_>::check )
				{
					for( unsigned k=0; k<6; ++k ){
						ff.read<int>( std::back_inserter(itmp) );
						for( unsigned j=0; j<itmp.size(); ++j )
							currlvl[j+gridoff].m_neighbour[k] = m_ind_grid_map[ (itmp[j]-m_ncoarse)%m_header.ngridmax ];
						itmp.clear();
					}
				}else
					ff.skip_n( 6 );
					
			
				//... son cell indices
				for( unsigned ind=0; ind<8; ++ind ){
					ff.read<int>( std::back_inserter(itmp) );
					for( unsigned k=0; k<itmp.size(); ++k ){
						currlvl[k+gridoff].m_son[ind] = m_ind_grid_map[ itmp[k] ];
					}
					itmp.clear();
				}

				//.. skip cpu + refinement map
				ff.skip_n( 8+8 );
			}else{
				//...skip entire record
				ff.skip_n( 3+3+1+6+8+8+8 );		
			}
		}
	}
}

} //namespace AMR
} //namespace ARTIO



#undef FIX

#endif //__ARTIO_AMR_DATA_HH
