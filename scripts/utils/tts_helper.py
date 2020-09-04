import datetime
import os
import time

from google.cloud import texttospeech
from pygame import mixer


class TTSHelper:
    """ helper class for google TTS
    set the environment variable GOOGLE_APPLICATION_CREDENTIALS first
    GOOGLE_APPLICATION_CREDENTIALS = 'path to json key file'
    """

    cache_folder = './cached_wav/'

    def __init__(self, cache_path=None):
        if cache_path is not None:
            self.cache_folder = cache_path

        # create cache folder
        try:
            os.makedirs(self.cache_folder)
        except OSError:
            pass

        # init tts
        self.client = texttospeech.TextToSpeechClient()
        self.voice_en_female = texttospeech.types.VoiceSelectionParams(
            language_code='en-US', name='en-US-Wavenet-F')
        self.voice_en_male = texttospeech.types.VoiceSelectionParams(
            language_code='en-US', name='en-US-Wavenet-D')
        self.audio_config_en = texttospeech.types.AudioConfig(
            speaking_rate=1.0,
            audio_encoding=texttospeech.enums.AudioEncoding.LINEAR16)

        # init player
        mixer.init()

        # clean up cache folder
        self._cleanup_cachefolder()

    def _cleanup_cachefolder(self):
        """ remove least accessed files in the cache """
        dir_to_search = self.cache_folder
        for dirpath, dirnames, filenames in os.walk(dir_to_search):
            for file in filenames:
                curpath = os.path.join(dirpath, file)
                file_accessed = datetime.datetime.fromtimestamp(os.path.getatime(curpath))
                if datetime.datetime.now() - file_accessed > datetime.timedelta(days=30):
                    os.remove(curpath)

    def _string2numeric_hash(self, text):
        import hashlib
        return int(hashlib.md5(text.encode('utf-8')).hexdigest()[:16], 16)

    def synthesis(self, ssml_text, voice_name='en-female', verbose=False):
        if not ssml_text.startswith(u'<speak>'):
            ssml_text = u'<speak>' + ssml_text + u'</speak>'

        filename = os.path.join(self.cache_folder, str(self._string2numeric_hash(voice_name + ssml_text)) + '.wav')

        # load or synthesis audio
        if not os.path.exists(filename):
            if verbose:
                start = time.time()

            # let's synthesis
            if voice_name == 'en-female':
                voice = self.voice_en_female
                audio_config = self.audio_config_en
            elif voice_name == 'en-male':
                voice = self.voice_en_male
                audio_config = self.audio_config_en
            else:
                raise ValueError

            synthesis_input = texttospeech.types.SynthesisInput(ssml=ssml_text)
            response = self.client.synthesize_speech(synthesis_input, voice, audio_config)

            if verbose:
                print('TTS took {0:.2f} seconds'.format(time.time() - start))
                start = time.time()

            # save to a file
            with open(filename, 'wb') as out:
                out.write(response.audio_content)
                if verbose:
                    print('written to a file "{}"'.format(filename))
        else:
            if verbose:
                print('use the cached wav "{}"'.format(filename))

        return filename

    def get_sound_obj(self, filename):
        # play
        sound = mixer.Sound(filename)
        length = sound.get_length()

        return sound, length

    def play(self, sound):
        sound.play(loops=0)
