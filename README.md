# Gesture Generation from Trimodal Context

This is an official pytorch implementation of *Speech Gesture Generation from the Trimodal Context of Text, Audio, and Speaker Identity (SIGGRAPH Asia 2020)*. In this paper, we present an automatic gesture generation model that uses the multimodal context of speech text, audio, and speaker identity to reliably generate gestures. By incorporating a multimodal context and an adversarial training scheme, the proposed model outputs gestures that are human-like and that match with speech content and rhythm. We also introduce a new quantitative evaluation metric, called FGD, for gesture generation models.

### [PAPER](https://arxiv.org/abs/2009.02119) | [VIDEO](https://youtu.be/2nDaBHUWpC0)

![OVERVIEW](.github/overview.jpg)

## Environment

This repository is developed and tested on Ubuntu 18.04, Python 3.6+, and PyTorch 1.3+. On Windows, we only tested the synthesis step and worked fine. On PyTorch 1.5+, some warning appears due to read-only entries in LMDB ([related issue](https://github.com/pytorch/pytorch/issues/37581)).

## Quick Start

### Installation

1. Clone this repository:
   ```
   git clone https://github.com/ai4r/Gesture-Generation-from-Trimodal-Context.git
   ```

0. Install required python packages:
   ```
   pip install -r requirements.txt
   ```

0. Install Gentle for audio-transcript alignment. Download the source code from [Gentle github](https://github.com/lowerquality/gentle) and install the library via `install.sh`. And then, you can import gentle library by specifying the path to the library at `script/synthesize.py` line 27.


### Preparation

1. Download [the trained model](https://kaistackr-my.sharepoint.com/:u:/g/personal/zeroyy_kaist_ac_kr/EdLj1u3V031Jm0YVJvM_O48BUpw2pBedu7LzLBS0YCB7SA).

0. Download [the preprocessed TED dataset](https://kaistackr-my.sharepoint.com/:u:/g/personal/zeroyy_kaist_ac_kr/EYAPLf8Hvn9Oq9GMljHDTK4BRab7rl9hAOcnjkriqL8qSg) (16GB) and extract the ZIP file into `data/ted_dataset`. You can find out the details of the TED datset from [here](https://github.com/youngwoo-yoon/youtube-gesture-dataset), and please refer to the paper how we extended the existing TED dataset.

0. Setup [Google Cloud TTS](https://cloud.google.com/text-to-speech). You need to set the environment variable `GOOGLE_APPLICATION_CREDENTIALS`. Please see [the manual here](https://cloud.google.com/docs/authentication/getting-started). You can skip this step if you're not going to synthesize gesture from custom text.


### Synthesize from TED speech

Generate gestures from a clip in the TED testset: 

```
python scripts/synthesize.py from_db_clip [trained model path] [number of samples to generate]
```

You would run like this:

```
python scripts/synthesize.py from_db_clip output/train_multimodal_context/multimodal_context_checkpoint_best.bin 10
```

The first run takes several minutes to cache the datset. After that, it runs quickly.   
You can find synthesized results in `output/generation_results`. There are MP4, WAV, and PKL files for visualized output, audio, and pickled raw results, respectively. Speaker IDs are randomly selected for each generation. The following shows a sample MP4 file.

![Sample MP4](.github/sample.gif)


### Synthesize from custom text

Generate gestures from speech text. Speech audio is synthesized by Google Cloud TTS.

```
python scripts/synthesize.py from_text [trained model path] {en-male, en-female}
```

You could select a sample text or input a new text. Input text can be a plain text or [SSML markup text](https://cloud.google.com/text-to-speech/docs/ssml). The third argument in the above command is for selecting TTS voice. You might further tweak TTS in `utils/tts_help.py`.
 

## Training

Train the proposed model:
```
python scripts/train.py --config=config/multimodal_context.yml
```

And the baseline models as well:

```
python scripts/train.py --config=config/seq2seq.yml
python scripts/train.py --config=config/speech2gesture.yml
python scripts/train.py --config=config/joint_embed.yml 
```

Caching TED training set (`lmdb_train`) takes tens of minutes at your first run. Model checkpoints and sample results will be saved in subdirectories of `output` folder. Training the proposed model took about 8 h with a RTX 2080 Ti.

Note on reproducibility:  
unfortunately, we didn't fix a random seed, so you are not able to reproduce the same FGD in the paper. But, several runs with different random seeds mostly fell in a similar FGD range.

## Fréchet Gesture Distance (FGD)

To be updated.

## Blender Animation (from a generated PKL file)

Versions:

* Blender 2.79B 
* FFMPEG git-2020-05-10-fc99a24 

Assure the file path structure as following:


```

Blender file (poseRender.blend)
Codec (h264_in_MP4.py)
Data Folder 
├── A generated PKL file (.pkl)
└── Audio file (.wav)
```
- Place the codec file as the same path with `.blend` file.
- Confirm the `.pkl` and `.wav` files in the data folder. 


Open blender file and set configuration :



```
1. data_folder                                       ## Name of Data Folder 
2. render_dir                                        ## Name of Render Folder
3. target_file                                       ## * : for all files, Names of file : render a specific file
4. You can change the details of render setting 
  - resolution_percentage                            ## defulat is 100
  - render_video                                     ## True : render video, False : render image frames only
  - test_run                                         ## True : render 10 frames for the test, Fasle : render all frames
  - upsample
  - out_fps 
  - verbose 
  - etc

* you don't need to change [character_num="123"]
```

You can find the details `renderAnim.py` in blender file.

If you success the path and configuration set up, 
Press [Run Script] button and enjoy! 


you can see the render output as below.


![blender output](.github/ot.gif | width == 300)



## License

Please see `LICENSE.md`


## Citation

If you find our work useful in your research, please consider citing:

```
@article{Yoon2020Trimodal,
  title={Speech Gesture Generation from the Trimodal Context of Text, Audio, and Speaker Identity},
  author={Youngwoo Yoon and Bok Cha and Joo-Haeng Lee and Minsu Jang and Jaeyeon Lee and Jaehong Kim and Geehyuk Lee},
  journal={ACM Transactions on Graphics},
  year={2020},
  volume={39},
  number={6},
}
```

Please feel free to contact us (youngwoo@etri.re.kr) with any question or concerns.
