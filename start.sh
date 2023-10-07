sudo apt-get update
sudo apt-get install python3-venv
python3 -m venv myenv
source myenv/bin/activate
pip install jupyter ipykernel
python -m ipykernel install --user --name=myenv

# ssh-keygen
