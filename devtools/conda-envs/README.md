### Notes

> conda activate espfit  
> conda env export --from-history > test_env.yaml  
> conda env create -f test_env.yaml -n test_env 
> conda uninstall --force openff-toolkit  
> pip install git+https://github.com/kntkb/openff-toolkit.git@7e9d0225782ef723083407a1cbf1f4f70631f934  
> #install espfit  
> pip install .  