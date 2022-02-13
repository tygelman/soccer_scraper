#!/bin/bash

#Get path to this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"


conda env update --file './envs/mac/flask.yml'


