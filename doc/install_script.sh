#
# Hi there!  Welcome to the yt installation script.
#
# This script is designed to create a fully isolated Python installation
# with the dependencies you need to run yt.
#
# There are a few options, but you only need to set *one* of them.  And
# that's the next one, DEST_DIR.  But, if you want to use an existing HDF5
# installation you can set HDF5_DIR, or if you want to use some other
# subversion checkout of YT, you can set YT_DIR, too.  (It'll already
# check the current directory and one up.
#
# NOTE: If you have trouble with wxPython, set INST_WXPYTHON=0 .
#
# And, feel free to drop me a line: matthewturk@gmail.com
#

DEST_DIR="`pwd`/yt-`uname -p`"   # Installation location

# Here's where you put the HDF5 path if you like; otherwise it'll download it
# and install it on its own
#HDF5_DIR=

# If you need to supply arguments to the NumPy build, supply them here
# This one turns on gfortran manually:
#NUMPY_ARGS="--fcompiler=gnu95"
# If you absolutely can't get the fortran to work, try this:
#NUMPY_ARGS="--fcompiler=fake"

INST_WXPYTHON=1 # If you 't want to install wxPython, set this to 1
INST_ZLIB=1     # On some systems (Kraken) matplotlib has issues with 
                # the system zlib, which is compiled statically.
                # If need be, you can turn this off.
INST_TRAITS=1   # Experimental TraitsUI installation

# If you've got YT some other place, set this to point to it.
YT_DIR=""

#------------------------------------------------------------------------------#
#                                                                              #
# Okay, the script starts here.  Feel free to play with it, but hopefully      #
# it'll work as is.                                                            #
#                                                                              #
#------------------------------------------------------------------------------#

function do_exit
{
    echo "Failure.  Check ${LOG_FILE}."
    exit 1
}

function do_setup_py
{
    [ -e $1/done ] && return
    echo "Installing $1 (arguments: '$*')"
    [ ! -e $1 ] && tar xfz $1.tar.gz
    cd $1
    shift
    ( ${DEST_DIR}/bin/python2.6 setup.py build   $* 2>&1 ) 1>> ${LOG_FILE} || do_exit
    ( ${DEST_DIR}/bin/python2.6 setup.py install    2>&1 ) 1>> ${LOG_FILE} || do_exit
    touch done
    cd ..
}

function get_enzotools
{
    echo "Downloading $1 from yt.enzotools.org"
    [ -e $1 ] && return
    wget -nv "http://yt.enzotools.org/dependencies/$1" || do_exit
    wget -nv "http://yt.enzotools.org/dependencies/$1.md5" || do_exit
    ( which md5sum &> /dev/null ) || return # return if we don't have md5sum
    ( md5sum -c $1.md5 2>&1 ) 1>> ${LOG_FILE} || do_exit
}

ORIG_PWD=`pwd`

LOG_FILE="${DEST_DIR}/yt_install.log"

if [ -z "${DEST_DIR}" ]
then
    echo "Edit this script, set the DEST_DIR parameter and re-run."
    exit 1
fi

echo "Installing into ${DEST_DIR}"
echo "INST_WXPYTHON=${INST_WXPYTHON}"

mkdir -p ${DEST_DIR}/src
cd ${DEST_DIR}/src

# Individual processes
if [ -z "$HDF5_DIR" ]
then
    echo "Downloading HDF5"
    get_enzotools hdf5-1.6.8.tar.gz
fi

[ $INST_ZLIB -eq 1 ] && get_enzotools zlib-1.2.3.tar.bz2 
[ $INST_WXPYTHON -eq 1 ] && get_enzotools wxPython-src-2.8.9.1.tar.bz2
get_enzotools Python-2.6.1.tgz
get_enzotools numpy-1.2.1.tar.gz
get_enzotools matplotlib-0.98.5.2.tar.gz
get_enzotools ipython-0.9.1.tar.gz
get_enzotools tables-2.1.tar.gz

if [ -z "$YT_DIR" ]
then
    if [ -e $ORIG_PWD/yt/mods.py ]
    then
        YT_DIR="$ORIG_PWD"
    elif [ -e $ORIG_PWD/../yt/mods.py ]
    then
        YT_DIR=`dirname $ORIG_PWD`
    elif [ ! -e yt-trunk-svn ] 
    then
        ( svn co http://svn.enzotools.org/yt/trunk/ ./yt-trunk-svn 2>&1 ) 1>> ${LOG_FILE}
        YT_DIR="$PWD/yt-trunk-svn/"
    elif [ -e yt-trunk-svn ] 
    then
        YT_DIR="$PWD/yt-trunk-svn/"
    fi
    echo Setting YT_DIR=${YT_DIR}
fi

if [ $INST_ZLIB -eq 1 ]
then
    if [ ! -e zlib-1.2.3/done ]
    then
        [ ! -e zlib-1.2.3 ] && tar xfj zlib-1.2.3.tar.bz2
        echo "Installing ZLIB"
        cd zlib-1.2.3
        ( ./configure --shared --prefix=${DEST_DIR}/ 2>&1 ) 1>> ${LOG_FILE} || do_exit
        ( make install 2>&1 ) 1>> ${LOG_FILE} || do_exit
        touch done
        cd ..
    fi
    ZLIB_DIR=${DEST_DIR}
    export LDFLAGS="${LDFLAGS} -L${ZLIB_DIR}/lib/ -L${ZLIB_DIR}/lib64/"
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${ZLIB_DIR}/lib/"
fi

if [ -z "$HDF5_DIR" ]
then
    if [ ! -e hdf5-1.6.8/done ]
    then
        [ ! -e hdf5-1.6.8 ] && tar xfz hdf5-1.6.8.tar.gz
        echo "Installing HDF5"
        cd hdf5-1.6.8
        ( ./configure --prefix=${DEST_DIR}/ --enable-shared 2>&1 ) 1>> ${LOG_FILE} || do_exit
        ( make install 2>&1 ) 1>> ${LOG_FILE} || do_exit
        touch done
        cd ..
    fi
    export HDF5_DIR=${DEST_DIR}
fi

if [ ! -e Python-2.6.1/done ]
then
    echo "Installing Python.  This may take a while, but don't worry.  YT loves you."
    [ ! -e Python-2.6.1 ] && tar xfz Python-2.6.1.tgz
    cd Python-2.6.1
    ( ./configure --prefix=${DEST_DIR}/ 2>&1 ) 1>> ${LOG_FILE} || do_exit

    ( make 2>&1 ) 1>> ${LOG_FILE} || do_exit
    ( make install 2>&1 ) 1>> ${LOG_FILE} || do_exit
    touch done
    cd ..
fi

export PYTHONPATH=${DEST_DIR}/lib/python2.6/site-packages/

if [ $INST_WXPYTHON -eq 1 ] && [ ! -e wxPython-src-2.8.9.1/done ]
then
    echo "Installing wxPython.  This may take a while, but don't worry.  YT loves you."
    [ ! -e wxPython-src-2.8.9.1 ] && tar xfj wxPython-src-2.8.9.1.tar.bz2
    cd wxPython-src-2.8.9.1

    ( ./configure --prefix=${DEST_DIR}/ --with-opengl 2>&1 ) 1>> ${LOG_FILE} || do_exit
    ( make install 2>&1 ) 1>> ${LOG_FILE} || do_exit
    cd contrib
    ( make install 2>&1 ) 1>> ${LOG_FILE} || do_exit
    cd ../wxPython/
    ( ${DEST_DIR}/bin/python2.6 setup.py WX_CONFIG=${DEST_DIR}/bin/wx-config install 2>&1 ) 1>> ${LOG_FILE} || do_exit

    touch ../done
    cd ../..
fi

export LDFLAGS="${LDFLAGS} -L${DEST_DIR}/lib/ -L${DEST_DIR}/lib64/"

echo "Installing setuptools"
( ${DEST_DIR}/bin/python2.6 ${YT_DIR}/ez_setup.py 2>&1 ) 1>> ${LOG_FILE} || do_exit

do_setup_py numpy-1.2.1 ${NUMPY_ARGS}
do_setup_py matplotlib-0.98.5.2
do_setup_py ipython-0.9.1
do_setup_py tables-2.1 

echo "Doing yt update"
MY_PWD=`pwd`
cd $YT_DIR
( svn up 2>&1 ) 1>> ${LOG_FILE}

echo "Installing yt"
echo $HDF5_DIR > hdf5.cfg
( ${DEST_DIR}/bin/python2.6 setup.py develop 2>&1 ) 1>> ${LOG_FILE} || do_exit
touch done
cd $MY_PWD

if [ $INST_WXPYTHON -eq 1 ] && [ $INST_TRAITS -eq 1 ]
then
    echo "Installing Traits"
    ( ${DEST_DIR}/bin/easy_install-2.6 -U TraitsGUI TraitsBackendWX 2>&1 ) 1>> ${LOG_FILE} || do_exit
fi

echo
echo
echo "========================================================================"
echo
echo "yt is now installed in $DEST_DIR ."
echo "To run from this new installation, the a few variables need to be"
echo "prepended with the following information:"
echo
echo "PATH            => $DEST_DIR/bin/"
echo "PYTHONPATH      => $DEST_DIR/lib/python2.6/site-packages/"
echo "LD_LIBRARY_PATH => $DEST_DIR/lib/"
echo
echo "For interactive data analysis and visualization, we recommend running"
echo "the IPython interface, which will become more fully featured with time:"
echo
echo "$DEST_DIR/bin/iyt"
echo
echo "For command line analysis run:"
echo
echo "$DEST_DIR/bin/yt"
echo
echo "Note of interest: this installation will use the directory"
echo "$YT_DIR"
echo "as the source for all the YT code.  This means you probably shouldn't"
echo "delete it, but on the plus side, any changes you make there are"
echo "automatically propagated."
echo
echo "For support, see one of the following websites:"
echo
echo "    http://yt.enzotools.org/wiki/"
echo "    http://yt.enzotools.org/doc/"
echo
echo "Or join the mailing list:"
echo 
echo "    http://lists.spacepope.org/listinfo.cgi/yt-users-spacepope.org"
echo
echo "========================================================================"
