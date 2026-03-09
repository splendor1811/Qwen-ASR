from src.data.processors.base import BaseProcessor
from src.data.processors.vivos import VIVOSProcessor
from src.data.processors.vlsp import VLSPProcessor
from src.data.processors.fleurs import FLEURSProcessor
from src.data.processors.gigaspeech2 import GigaSpeech2Processor
from src.data.processors.vietbud500 import VietBud500Processor
from src.data.processors.vietsuperspeech import VietSuperSpeechProcessor
from src.data.processors.fosd import FOSDProcessor
from src.data.processors.phoaudiobook import PhoAudioBookProcessor
from src.data.processors.vivoice import ViVoiceProcessor

PROCESSOR_REGISTRY: dict[str, type[BaseProcessor]] = {
    "vivos": VIVOSProcessor,
    "vlsp": VLSPProcessor,
    "fleurs": FLEURSProcessor,
    "gigaspeech2": GigaSpeech2Processor,
    "vietbud500": VietBud500Processor,
    "vietsuperspeech": VietSuperSpeechProcessor,
    "fosd": FOSDProcessor,
    "phoaudiobook": PhoAudioBookProcessor,
    "vivoice": ViVoiceProcessor,
}
