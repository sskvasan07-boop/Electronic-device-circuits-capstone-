import train_model as tm, os

folder = r'd:\Electronic device and circuits notes\OPTOSCAN UI'
csvs = [os.path.join(folder, f) for f in sorted(os.listdir(folder)) if f.endswith('.csv')]
print(f'Training on {len(csvs)} CSV files:')
for c in csvs:
    print(' -', os.path.basename(c))

results = tm.train_and_save_merged(csvs, os.path.join(folder, 'trained_model.pkl'))

print()
print(f'Accuracy : {results["accuracy"]*100:.2f}%')
print(f'Classes  : {results["class_names"]}')
print(f'Samples  : {results["n_samples"]} ({results["n_train"]} train / {results["n_test"]} test)')
print(f'Features : {results["n_features"]} wavelength bands')
print(f'Saved to : {results["model_path"]}')
