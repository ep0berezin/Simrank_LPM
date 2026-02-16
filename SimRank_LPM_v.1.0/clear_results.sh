#!/bin/bash

find results/data -mindepth 1 ! -name '.gitkeep' -exec rm -rf {} +

find results/img -mindepth 1 ! -name '.gitkeep' -exec rm -rf {} +

find results/log -mindepth 1 ! -name '.gitkeep' -exec rm -rf {} +

