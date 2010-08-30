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
# And, feel free to drop me a line: matthewturk@gmail.com
#

DEST_SUFFIX="yt-`uname -p`"
DEST_DIR="`pwd`/${DEST_SUFFIX/ /}"   # Installation location
PY_DIR="/Library/Frameworks/Python.framework/Versions/Current/"

# Here's where you put the HDF5 path if you like; otherwise it'll download it
# and install it on its own
HDF5_DIR=

INST_HG=1       # Install Mercurial or not?
INST_GUI=1      # Install the necessary bits for the GUI?
# If you've got YT some other place, set this to point to it.
YT_DIR=

#------------------------------------------------------------------------------#
#                                                                              #
# Okay, the script starts here.  Feel free to play with it, but hopefully      #
# it'll work as is.                                                            #
#                                                                              #
#------------------------------------------------------------------------------#

shopt -s extglob

function do_exit
{
    echo "Failure.  Check ${LOG_FILE}."
    exit 1
}

function do_setup_py
{
    [ -e $1/done ] && return
    echo
    echo "Installing $1 (may need sudo)"
    echo
    [ ! -e $1 ] && tar xfz $1.tar.gz
    cd $1
    if [ ! -z `echo $1 | grep h5py` ]
    then
	echo "${PY_DIR}/bin/python2.6 setup.py configure --hdf5=${HDF5_DIR}"
	( ${PY_DIR}/bin/python2.6 setup.py configure --hdf5=${HDF5_DIR} 2>&1 ) 1>> ${LOG_FILE} || do_exit
    fi
    shift
    ( sudo ${PY_DIR}/bin/python2.6 setup.py install $* 2>&1 ) 1>> ${LOG_FILE} || do_exit
    touch done
    cd ..
}

function get_enzotools
{
    echo "Downloading $1 from yt.enzotools.org"
    [ -e $1 ] && return
    curl "http://yt.enzotools.org/dependencies/osx/$1" -o $1 || do_exit
    curl "http://yt.enzotools.org/dependencies/osx/$1.md5" -o $1.md5 || do_exit
    ( which md5sum &> /dev/null ) || return # return if we don't have md5sum
    ( md5sum -c $1.md5 2>&1 ) 1>> ${LOG_FILE} || do_exit
}

function self_install
{
    echo 
    echo "--------------------------------------------------------------------------------"
    echo "Installing ${1}.  You will need to handle this procedure."
    echo
    echo "Press enter to start, then return when finished."
    echo
    echo "--------------------------------------------------------------------------------"
    echo
    read LOKI
    ext="${1##*.}"
    if [ "${ext}" = "dmg" ] 
    then
        [ ! -d ${DEST_DIR}/src/mount_point/ ] && \
            ( mkdir ${DEST_DIR}/src/mount_point/ 2>&1 ) >> ${LOG_FILE}
        ( hdiutil unmount ${DEST_DIR}/src/mount_point 2>&1 ) >> ${LOG_FILE}
        ( hdiutil mount ${1} -mountpoint ${DEST_DIR}/src/mount_point/ 2>&1 ) \
            >> ${LOG_FILE}
        open ${DEST_DIR}/src/mount_point/?(*.mpkg|*.pkg)
    else
        open ${1}
    fi
    echo
    echo "--------------------------------------------------------------------------------"
    echo
    echo "Press enter when the installation is complete."
    echo
    echo "--------------------------------------------------------------------------------"
    echo
    read LOKI
    [ "${ext}" = "dmg" ] && \
        ( hdiutil unmount ${DEST_DIR}/src/mount_point/ 2>&1 ) >> ${LOG_FILE}
}

ORIG_PWD=`pwd`

LOG_FILE="${DEST_DIR}/yt_install.log"

if [ -z "${DEST_DIR}" ]
then
    echo "Edit this script, set the DEST_DIR parameter and re-run."
    exit 1
fi

echo "Installing into ${DEST_DIR}"

mkdir -p ${DEST_DIR}/src
cd ${DEST_DIR}/src

# Individual processes
if [ -z "$HDF5_DIR" ]
then
    echo "Downloading HDF5"
    get_enzotools hdf5-1.6.9.tar.gz
fi

get_enzotools python-2.6.4-macosx10.6-2010-02-01.dmg
get_enzotools numpy-1.3.0-py2.6-python.org.dmg
get_enzotools matplotlib-0.99.1.2-py2.6-macosx10.6.dmg
get_enzotools ipython-0.10.tar.gz
get_enzotools h5py-1.2.0.tar.gz

if [ -z "$YT_DIR" ]
then
    if [ -e $ORIG_PWD/yt/mods.py ]
    then
        YT_DIR="$ORIG_PWD"
    elif [ -e $ORIG_PWD/../yt/mods.py ]
    then
        YT_DIR=`dirname $ORIG_PWD`
    elif [ ! -e yt-1.7-svn ] 
    then
        ( svn co http://svn.enzotools.org/yt/branches/yt-1.7/ ./yt-1.7-svn 2>&1 ) 1>> ${LOG_FILE}
        YT_DIR="$PWD/yt-1.7-svn/"
    elif [ -e yt-1.7-svn ] 
    then
        YT_DIR="$PWD/yt-1.7-svn/"
    fi
    echo Setting YT_DIR=${YT_DIR}
fi

if [ -z "$HDF5_DIR" ]
then
    if [ ! -e hdf5-1.6.9/done ]
    then
        [ ! -e hdf5-1.6.9 ] && tar xfz hdf5-1.6.9.tar.gz
        echo "Installing HDF5"
        cd hdf5-1.6.9
        ( ./configure --prefix=${DEST_DIR}/ --enable-shared 2>&1 ) 1>> ${LOG_FILE} || do_exit
        ( make install 2>&1 ) 1>> ${LOG_FILE} || do_exit
        touch done
        cd ..
    fi
    export HDF5_DIR=${DEST_DIR}
else
    export HDF5_DIR=${HDF5_DIR}
fi
export HDF5_API=16

[ ! -e ${DEST_DIR}/src/py_done ] && self_install \
    python-2.6.4-macosx10.6-2010-02-01.dmg
touch ${DEST_DIR}/src/py_done

echo "Installing distribute (needs sudo)"
echo
( sudo ${PY_DIR}/bin/python2.6 ${YT_DIR}/distribute_setup.py 2>&1 ) 1>> ${LOG_FILE} || do_exit

echo "Installing pip"
( sudo ${PY_DIR}/bin/easy_install-2.6 pip 2>&1 ) 1>> ${LOG_FILE} || do_exit

echo "Installing readline"
( sudo ${PY_DIR}/bin/easy_install-2.6 readline 2>&1 ) 1>> ${LOG_FILE} || do_exit

[ ! -e ${DEST_DIR}/src/np_done ] && self_install \
    numpy-1.3.0-py2.6-python.org.dmg
touch ${DEST_DIR}/src/np_done

[ ! -e ${DEST_DIR}/src/mp_done ] && self_install \
    matplotlib-0.99.1.2-py2.6-macosx10.6.dmg
touch ${DEST_DIR}/src/mp_done

echo "Installing pytz"
( sudo ${PY_DIR}/bin/easy_install-2.6 pytz 2>&1 ) 1>> ${LOG_FILE} || do_exit

do_setup_py ipython-0.10
do_setup_py h5py-1.2.0

echo "Doing yt update"
MY_PWD=`pwd`
cd $YT_DIR
( svn up 2>&1 ) 1>> ${LOG_FILE}

echo "Installing yt (may need sudo)"
echo $HDF5_DIR > hdf5.cfg
echo /usr/X11 > png.cfg # I think this should work everywhere
( ${PY_DIR}/bin/python2.6 setup.py build_ext -i 2>&1 ) 1>> ${LOG_FILE} || do_exit
( sudo ${PY_DIR}/bin/python2.6 setup.py develop 2>&1 ) 1>> ${LOG_FILE} || do_exit
touch done
cd $MY_PWD

if [ $INST_HG -eq 1 ]
then
    echo "Installing Mercurial."
    ( sudo ${PY_DIR}/bin/pip install -U mercurial 2>&1 ) 1>> ${LOG_FILE} || do_exit
fi


if [ $INST_GUI -eq 1 ]
then
    # This is a long, long list.
    get_enzotools qt-mac-cocoa-opensource-4.6.2.dmg 
    get_enzotools VTK-5.5.0-Darwin.dmg
    get_enzotools VTK-5.5-Darwin-Python.dmg
    get_enzotools ETS-20100316.tar.gz
    get_enzotools sip-4.9.1.tar.gz
    get_enzotools PyQt-mac-gpl-4.6.1.tar.gz

    # We do these in order!  Hooray!
    [ ! -e ${DEST_DIR}/src/qt_done ] && self_install \
        qt-mac-cocoa-opensource-4.6.2.dmg 
    touch ${DEST_DIR}/src/qt_done

    [ ! -e ${DEST_DIR}/src/vtk_done ] && self_install \
        VTK-5.5.0-Darwin.dmg
    touch ${DEST_DIR}/src/vtk_done

    [ ! -e ${DEST_DIR}/src/pyvtk_done ] && self_install \
        VTK-5.5-Darwin-Python.dmg
    touch ${DEST_DIR}/src/pyvtk_done

    if [ ! -e ${DEST_DIR}/src/sip-4.9.1/done ]
    then
        [ ! -e sip-4.9.1 ] && tar xvfz sip-4.9.1.tar.gz
        echo "Installing SIP."
        echo "You may be asked to accept a license, and this may require sudo."
        cd sip-4.9.1
        ( ${PY_DIR}/bin/python2.6 configure.py 0>&0 2>&1 ) 1>> ${LOG_FILE} || do_exit
        ( make 2>&1 ) 1>> ${LOG_FILE} || do_exit
        ( sudo make install 2>&1 ) 1>> ${LOG_FILE} || do_exit
        touch done
        cd ..
    fi

    if [ ! -e ${DEST_DIR}/src/PyQt-mac-gpl-4.6.1/done ]
    then
        [ ! -e PyQt-mac-gpl-4.6.1 ] && tar xvfz PyQt-mac-gpl-4.6.1.tar.gz
        echo
        echo "Installing PyQt4."
        echo "You may be asked to accept a license, and this may require sudo."
        echo "Once you've answered, the compilation may take a while.  You could get a sandwich."
        echo
        cd PyQt-mac-gpl-4.6.1
        ( ${PY_DIR}/bin/python2.6 configure.py --qmake=/usr/bin/qmake 0>&0 2>&1 ) | tee -a ${LOG_FILE} || do_exit
        ( make 2>&1 ) 1>> ${LOG_FILE} || do_exit
        ( sudo make install 2>&1 ) 1>> ${LOG_FILE} || do_exit
        touch done
        cd ..
    fi

    if [ ! -e ${DEST_DIR}/src/ets_done ] 
    then
        echo "Installing ETS (needs sudo)"
        tar xvfz ETS-20100316.tar.gz
        sudo easy_install -N ETS_3.3.1/dist/*.egg
    fi
    touch ${DEST_DIR}/src/ets_done

    echo "GUI installed successfully."
fi

echo
echo
echo "========================================================================"
echo
echo "yt is now installed in $DEST_DIR ."
echo "To run from this new installation, the a few variables need to be"
echo "prepended with the following information:"
echo
echo "PATH => $PY_DIR/bin/"
echo
echo "For interactive data analysis and visualization, we recommend running"
echo "the IPython interface, which will become more fully featured with time:"
echo
echo "$PY_DIR/bin/iyt"
echo
echo "For command line analysis run:"
echo
echo "$PY_DIR/bin/yt"
echo
echo "Note of interest: this installation will use the directory"
echo "$YT_DIR"
echo "as the source for all the YT code.  This means you probably shouldn't"
echo "delete it, but on the plus side, any changes you make there are"
echo "automatically propagated."
if [ $INST_HG -eq 1 ]
then
  echo "Mercurial has also been installed:"
  echo
  echo "$DEST_DIR/bin/hg"
  echo
fi
if [ $INST_GUI -eq 1 ]
then
  echo "The GUI toolkit dependencies have also been installed."
  echo "You will need to add /usr/lib/vtk-5.5/ to your DYLD_LIBRARY_PATH ."
  echo "Additionally, set ETS_TOOLKIT to 'qt4' in your environment."
fi
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
echo
echo "      Everything is fine.  Nothing is ruined."
echo
echo "========================================================================"
