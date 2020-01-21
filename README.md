# UNet: semantic segmentation with PyTorch

The code was modified from GitHub’s code:
[https://github.com/milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet)
After downloading the code, there are some problems with compiling, so I made some modifications. I modified the test that just put images data into the folder instead of inputting image names when compiling the prediction code. Besides, I added the calculating dice coefficient function to evaluate the testing result.

## Data:
The data used here is the original data that Kaggle’s Carvana Image Mask Challenge from high definition images. And the Carvana data is available on the [Kaggle’s website](https://www.kaggle.com/c/carvana-image-masking-challenge/data).

## Training:

```shell script
> python train.py -h
usage: train.py [-h] [-e E] [-b [B]] [-l [LR]] [-f LOAD] [-s SCALE] [-v VAL]

optional arguments:
  -h, --help            show this help message and exit
  -e E, --epochs E      Number of epochs (default: 5)
  -b [B], --batch-size [B]
                        Batch size (default: 1)
  -l [LR], --learning-rate [LR]
                        Learning rate (default: 0.1)
  -f LOAD, --load LOAD  Load model from a .pth file (default: False)
  -s SCALE, --scale SCALE
                        Downscaling factor of the images (default: 0.5)
  -v VAL, --validation VAL
                        Percent of the data that is used as validation (0-100)
                        (default: 15.0)
```

The default scale is 0.5. The original developer advises to set it to 1. The input images and masks should be in the data folder, and the trained model will be in the checkpoints folder.

## Prediction:
To predict the images and show them without saved:
`python predict.py –m model.pth --viz --no-save`

```shell script
> python predict.py -h
usage: predict.py [-h] [--model FILE] --input INPUT [INPUT ...]
                  [--output INPUT [INPUT ...]] [--viz] [--no-save]
                  [--mask-threshold MASK_THRESHOLD] [--scale SCALE]

Predict masks from input images

optional arguments:
  -h, --help            show this help message and exit
  --model FILE, -m FILE
                        Specify the file in which the model is stored
  --output INPUT [INPUT ...], -o INPUT [INPUT ...]
                        Filenames of ouput images (default: None)
  --viz, -v             Visualize the images as they are processed (default:
                        False)
  --no-save, -n         Do not save the output masks (default: False)
  --mask-threshold MASK_THRESHOLD, -t MASK_THRESHOLD
                        Minimum probability value to consider a mask pixel
                        white (default: 0.5)
  --scale SCALE, -s SCALE
                        The scale factor for the input images (default: 0.5)
```

There is no the default model. Move the trained model into the Unet-Pytorch folder. The input testing images and masks should be in the test folder. The final result mask will be in the result folder.

-------------------------------------------------------------------------------

這裡提供的程式碼是修改GitHub上的程式碼：
[https://github.com/milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet)
由於下載後，在使用上有些問題，因此做了一些修改。並且將原本測試時，需要輸入測試的圖片名稱改成，只需將影像放入資料夾內，即可做大量測試。另外，加入了dice coefficient，來評估測試的結果。

## 資料:
這裡使用的資料，跟原本的來源相同。都是高解析度的Kaggle's Carvana Image Masking Challenge資料。Carvana 的資料可以在[Kaggle’s的網站](https://www.kaggle.com/c/carvana-image-masking-challenge/data)上找到。

## 訓練:

```shell script
> python train.py -h
usage: train.py [-h] [-e E] [-b [B]] [-l [LR]] [-f LOAD] [-s SCALE] [-v VAL]

optional arguments:
  -h, --help            show this help message and exit
  -e E, --epochs E      Number of epochs (default: 5)
  -b [B], --batch-size [B]
                        Batch size (default: 1)
  -l [LR], --learning-rate [LR]
                        Learning rate (default: 0.1)
  -f LOAD, --load LOAD  Load model from a .pth file (default: False)
  -s SCALE, --scale SCALE
                        Downscaling factor of the images (default: 0.5)
  -v VAL, --validation VAL
                        Percent of the data that is used as validation (0-100)
                        (default: 15.0)
```

不改scale值，默認是0.5。但原本的開發者建議是使用1。
使用方法是將影像與mask分別放到data的資料夾內。
訓練出來的模型會在checkpoints資料夾內。

## 預測:
預測影像，顯示他，但不保存:
`python predict.py –m model.pth --viz --no-save`

```shell script
> python predict.py -h
usage: predict.py [-h] [--model FILE] --input INPUT [INPUT ...]
                  [--output INPUT [INPUT ...]] [--viz] [--no-save]
                  [--mask-threshold MASK_THRESHOLD] [--scale SCALE]

Predict masks from input images

optional arguments:
  -h, --help            show this help message and exit
  --model FILE, -m FILE
                        Specify the file in which the model is stored
  --output INPUT [INPUT ...], -o INPUT [INPUT ...]
                        Filenames of ouput images (default: None)
  --viz, -v             Visualize the images as they are processed (default:
                        False)
  --no-save, -n         Do not save the output masks (default: False)
  --mask-threshold MASK_THRESHOLD, -t MASK_THRESHOLD
                        Minimum probability value to consider a mask pixel
                        white (default: 0.5)
  --scale SCALE, -s SCALE
                        The scale factor for the input images (default: 0.5)
```

並沒有默認的模型。因此訓練完的模型可以拉到Unet-Pytorch資料夾內。
要測試的影像與mask，分別放到test資料夾內。
最後出來的mask會放到result內。

