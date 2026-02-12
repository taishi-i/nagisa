import nagisa_utils as utils

from nagisa.tagger import Tagger
from nagisa.train import fit
from nagisa.stopwords import stopwords

version = '0.2.12'
# Initialize instance
tagger  = Tagger()
# Functions
wakati  = tagger.wakati
tagging = tagger.tagging
filter  = tagger.filter
extract = tagger.extract
postagging = tagger.postagging
decode = tagger.decode
fit = fit
stopwords = stopwords

__version__ = version
