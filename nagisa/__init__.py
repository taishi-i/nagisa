import utils
from nagisa.tagger import Tagger
from nagisa.train import fit

version = '0.2.6'
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

__version__ = version
