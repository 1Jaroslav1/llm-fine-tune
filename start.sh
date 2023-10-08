sudo apt-get update
sudo apt-get install python3-venv
python3 -m venv myenv
source myenv/bin/activate
pip install jupyter ipykernel
python -m ipykernel install --user --name=myenv

# ssh-keygen
# rm -rf ~/.cache/*
# rm -rf ~/.local/share/Trash/info/* 
# rm -rf ~/.local/share/Trash/files/*
