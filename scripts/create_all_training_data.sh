script_path="/nfs/home/arafatj/master_project/src/create_train_data.py"

echo "python "$script_path" --speaker --seg --hierarchy 0"
python "$script_path" --speaker --seg --hierarchy 0
echo

echo "python "$script_path" --speaker --seg --hierarchy 1"
python "$script_path" --speaker --seg --hierarchy 1
echo



echo "python "$script_path" --speaker --sw --hierarchy 0"
python "$script_path" --speaker --sw --hierarchy 0
echo

echo "python "$script_path" --speaker --sw --hierarchy 1"
python "$script_path" --speaker --sw --hierarchy 1
echo



echo "python "$script_path" --situations --seg"
python "$script_path" --situations --seg
echo

echo "python "$script_path" --situations --sw"
python "$script_path" --situations --sw
echo