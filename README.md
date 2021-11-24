# Parallel passages

[![SWH](https://archive.softwareheritage.org/badge/origin/https://github.com/ETCBC/parallels/)](https://archive.softwareheritage.org/browse/origin/https://github.com/ETCBC/parallels/)
[![DOI](https://zenodo.org/badge/104842865.svg)](https://zenodo.org/badge/latestdoi/104842865)
[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)

[![etcbc](images/etcbc.png)](http://www.etcbc.nl)
[![dans](images/dans.png)](https://dans.knaw.nl/en)
[![tf](programs/images/tf-small.png)](https://annotation.github.io/text-fabric/tf)

### BHSA Family

* [bhsa](https://github.com/etcbc/bhsa) Core data and feature documentation
* [phono](https://github.com/etcbc/phono) Phonological representation of Hebrew words
* [parallels](https://github.com/etcbc/parallels) Links between similar verses
* [valence](https://github.com/etcbc/valence) Verbal valence for all occurrences
  of some verbs
* [trees](https://github.com/etcbc/trees) Tree structures for all sentences
* [bridging](https://github.com/etcbc/bridging) Open Scriptures morphology
  ported to the BHSA
* [pipeline](https://github.com/etcbc/pipeline) Generate the BHSA and SHEBANQ
  from internal ETCBC data files
* [shebanq](https://github.com/etcbc/shebanq) Engine of the
  [shebanq](https://shebanq.ancient-data.org) website

### Extended family

* [dss](https://github.com/etcbc/dss) Dead Sea Scrolls
* [extrabiblical](https://github.com/etcbc/extrabiblical)
  Extra-biblical writings from ETCBC-encoded texts
* [peshitta](https://github.com/etcbc/peshitta)
  Syriac translation of the Hebrew Bible
* [syrnt](https://github.com/etcbc/syrnt)
  Syriac translation of the New Testament

## About

Algorithm to determine parallel passages in the Hebrew Bible.

Part of the
[SYNVAR](https://www.nwo.nl/en/research-and-results/research-projects/i/30/9930.html)
project carried out at the 
[ETCBC](http://etcbc.nl)

## Results

The results of this study are being delivered in several forms, summarized here.

* **Case study**: a 
  [case study](https://github.com/ETCBC/parallels/blob/master/programs/kings_ii.ipynb)
  showing the parallels involved in 2 Kings 19-26; 
* **Annotations**: a set of
  [annotations](https://shebanq.ancient-data.org/hebrew/note?version=4b&id=Mnxjcm9zc3JlZg__&tp=txt_tb1&nget=v)
  in **SHEBANQ** showing the parallels as clickable cross-references;
* **Data module**: a set of 
  [higher level features](https://github.com/ETCBC/parallels/tree/master/tf)
  in **text-fabric** format, storing the cross-references as features;
* **Documentation**: a
  [Jupyter notebook](https://nbviewer.jupyter.org/github/etcbc/parallels/blob/master/programs/parallels.ipynb)
* **Program code**: a
  [Jupyter notebook](https://github.com/ETCBC/parallels/tree/master/programs)
  Codes the algorithm and applies it using many different
  parameter settings.

## Authors
* [Martijn Naaijer](mailto:m.naaijer@vu.nl) -
  [VU ETCBC](http://etcbc.nl) -
  Ph.D. student in Biblical Hebrew
* [Dirk Roorda](https://github.com/dirkroorda)
  [DANS](https://dans.knaw.nl/en/front-page?set_language=en) -
  author of the supporting library
  [Text-Fabric](https://github.com/Dans-labs/text-fabric).



