### What to do

- Install CUDA toolkit
  https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local

- Add to PATH

```
sudo chmod +w .bashrc
echo 'export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
source ~/.bashrc
```


- Verify

```
nvcc --version
```

- Install all requirements

```
python3 -m venv .venv
source ./.venv/bin/activate

pip3 install pandas torch transformers peft numpy requests parquet pyarrow datasets datetime bitsandbytes
pip3 install packaging ninja
pip3 install flash-attn --no-build-isolation
```

- Restart Kernel!
- Run(?)
