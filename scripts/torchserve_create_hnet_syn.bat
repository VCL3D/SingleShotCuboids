torch-model-archiver --model-name hnet_syn --version 1.0 ^
 -f --model-file ../hnet/model.py ^
 --serialized-file ../ckpt/hnet_syn.pth ^
 --handler ../hnet_handler.py --export-path ../mar ^
 --extra-files ../usdz_exporter.py,../obj_handler.py,../boundary_handler.py,requirements.txt
pause