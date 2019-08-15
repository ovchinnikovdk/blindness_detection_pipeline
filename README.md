# NN Pipeline for APTOS 2019 Blindness Detection

[APTOS 2019 Blindness Detection](https://www.kaggle.com/c/aptos2019-blindness-detection)

For training run:

```shell script
python train.py --config=json/path/to/cofig/file.json
```

Example for configuration:

```json
{
  "model" : {
    "name": "file.ClassName",
    "state_dict": "path/to/state_dict.dat",
    "params": {
      "in_channels" : 10,
      "out_channels" : 1,
      "num_layers" : 100,
      "etc" : "ok"
    }
  },
  "gpu" : true,
  "epochs" : 30,
  "batch_size" : 64,
  "data_path" : "../input/data_path"
}
```
