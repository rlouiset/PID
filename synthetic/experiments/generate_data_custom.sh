
python synthetic/generate_data_custom.py --num-data 20000 --setting redundancy --out-path synthetic/experiments
python synthetic/generate_data_custom.py --num-data 20000 --setting uniqueness0 --out-path synthetic/experiments
python synthetic/generate_data_custom.py --num-data 20000 --setting uniqueness1 --out-path synthetic/experiments
python synthetic/generate_data_custom.py --num-data 20000 --setting synergy --out-path synthetic/experiments
python synthetic/generate_data_custom.py --num-data 20000 --setting mix1 --mix-ratio 0.33 0 0 0.67 --out-path synthetic/experiments
python synthetic/generate_data_custom.py --num-data 20000 --setting mix2 --mix-ratio 0 0 0.67 0.33 --out-path synthetic/experiments
python synthetic/generate_data_custom.py --num-data 20000 --setting mix3 --mix-ratio 0.33 0.67 0 0 --out-path synthetic/experiments
python synthetic/generate_data_custom.py --num-data 20000 --setting mix4 --mix-ratio 0 0.33 0.67 0 --out-path synthetic/experiments
python synthetic/generate_data_custom.py --num-data 20000 --setting triplemix1 --mix-ratio 0.5 0.25 0.25 0 --out-path synthetic/experiments
python synthetic/generate_data_custom.py --num-data 20000 --setting triplemix2 --mix-ratio 0 0.25 0.5 0.25 --out-path synthetic/experiments
python synthetic/generate_data_custom.py --num-data 20000 --setting triplemix3 --mix-ratio 0.5 0 0.25 0.25 --out-path synthetic/experiments
python synthetic/generate_data_custom.py --num-data 20000 --setting triplemix4 --mix-ratio 0.25 0.25 0 0.5 --out-path synthetic/experiments
