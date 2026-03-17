#!/bin/bash
set -e

cd "$(dirname "$0")"

if [ ! -d "build" ]; then
    echo "Creating build directory..."
    mkdir build
fi

cd build

echo "Configuring with CMake..."
cmake ..

echo "Building..."
cmake --build . -j4

echo "Running..."
cd ..
./build/RealTimeCameraPipeline "$@"
