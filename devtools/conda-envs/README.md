### Notes

>conda activate espfit  
>conda env export --from-history > test_env.yaml  
>conda env create -f test_env.yaml -n test_env  
>#uninstall openff-toolkit and install a customized version to support dgl graphs created using openff-toolkit=0.10.6  
>conda uninstall --force openff-toolkit  
>pip install git+https://github.com/kntkb/openff-toolkit.git@7e9d0225782ef723083407a1cbf1f4f70631f934  
>#uninstall openmmforcefields if < 0.12.0  
>#use pip instead of mamba to avoid dependency issues with ambertools and python  
>conda uninstall --force openmmforcefields  
>pip install git+https://github.com/openmm/openmmforcefields@0.12.0  
>#install espfit  
>pip install .  
