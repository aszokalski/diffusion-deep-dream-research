#!/bin/bash

srun --partition=plgrid-gpu-a100 --nodes=1 --gres=gpu:0 --mem=32G --ntasks-per-node=1 --time=03:00:00 --pty bash -i