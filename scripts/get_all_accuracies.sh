script_path="/nfs/home/arafatj/master_project/src/training.py"
echo

# SPEAKER - Hierarchy 0
echo "--------------------------- SPEAKER - Hierarchy 0 ---------------------------"
echo "python "$script_path" --kfold --speaker --rf --seg --hierarchy 0"
python "$script_path" --kfold --speaker --rf --seg --hierarchy 0
echo

echo "python "$script_path" --kfold --speaker --rf --sw --hierarchy 0"
python "$script_path" --kfold --speaker --rf --sw --hierarchy 0
echo

echo "python "$script_path" --kfold --speaker --xgb --seg --hierarchy 0"
python "$script_path" --kfold --speaker --xgb --seg --hierarchy 0
echo

echo "python "$script_path" --kfold --speaker --xgb --sw --hierarchy 0"
python "$script_path" --kfold --speaker --xgb --sw --hierarchy 0
echo "-----------------------------------------------------------------------------"
echo

# SPEAKER - Hierarchy 1
echo "--------------------------- SPEAKER - Hierarchy 1 ---------------------------"
echo "python "$script_path" --kfold --speaker --rf --seg --hierarchy 1"
python "$script_path" --kfold --speaker --rf --seg --hierarchy 1
echo

echo "python "$script_path" --kfold --speaker --rf --sw --hierarchy 1"
python "$script_path" --kfold --speaker --rf --sw --hierarchy 1
echo

echo "python "$script_path" --kfold --speaker --xgb --seg --hierarchy 1"
python "$script_path" --kfold --speaker --xgb --seg --hierarchy 1
echo

echo "python "$script_path" --kfold --speaker --xgb --sw --hierarchy 1"
python "$script_path" --kfold --speaker --xgb --sw --hierarchy 1
echo "------------------------------------------------------------------------------"
echo

# SITUATIONS
echo "--------------------------------- SITUATIONS ---------------------------------"
echo "python "$script_path" --kfold --situations --rf --seg"
python "$script_path" --kfold --situations --rf --seg
echo

echo "python "$script_path" --kfold --situations --rf --sw"
python "$script_path" --kfold --situations --rf --sw
echo

echo "python "$script_path" --kfold --situations --xgb --seg"
python "$script_path" --kfold --situations --xgb --seg
echo

echo "python "$script_path" --kfold --situations --xgb --sw"
python "$script_path" --kfold --situations --xgb --sw
echo "------------------------------------------------------------------------------"
echo