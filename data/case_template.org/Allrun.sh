#!/bin/bash
cd ${0%/*} || exit 1    # Run from this directory

source /opt/openfoam9/etc/bashrc

# Source tutorial run functions
. $WM_PROJECT_DIR/bin/tools/RunFunctions

application=$(getApplication)

runApplication $application

#------------------------------------------------------------------------------
