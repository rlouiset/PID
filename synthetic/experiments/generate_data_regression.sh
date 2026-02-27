
python synthetic/generate_data_regression.py --num-data 20000 --setting redundancy --out-path synthetic/experiments
python synthetic/generate_data_regression.py --num-data 20000 --setting uniqueness0 --out-path synthetic/experiments
python synthetic/generate_data_regression.py --num-data 20000 --setting synergy --out-path synthetic/experiments
python synthetic/generate_data_regression.py --num-data 20000 --setting mix2 --mix-ratio 0 0 0.33 0.67 --out-path synthetic/experiments
