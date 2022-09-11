# Search engine

This search engine is based on a vector representation of documents, where values are calculated via **tf_idf** (two ways of counting tf: **count** and **log(1+count)**)

Documents are sentences from a collection of texts.  
The texts in the [`/text`](https://github.com/vsha96/search_system/tree/main/text) are taken from links to interesting Wikipedia facts.

When searching by query, the query is also translated into the vector space of documents, and the system outputs relevant documents as they are close (proximity is considered as the cosine of the angle between vectors)

The system works with Russian, but English language processing is also possible.

The algorithm operation is described in detail in the [`search_sys.ipynb`](https://github.com/vsha96/search_system/blob/main/search_sys.ipynb) (RU)

## Installation
- clone this repo  
`git clone git@github.com:vsha96/search_system.git`
- import the module
```
import mysearchsys as mss
```



## Usage

- The procedure for building a collection from files in [`/text`](https://github.com/vsha96/search_system/tree/main/text)  
`mss.make_collection()`
- Collection search function, returns a tuple:  
  **(** *list of docs when tf = count* **,** *list of docs when tf = log(1+count)* **)**  
`mss.search('your query')`

See examples in the [`main.py`](https://github.com/vsha96/search_system/blob/main/main.py)



## Features
- Dates, Roman numerals, English names are also terms (this may play a role when searching for a document)
- It is possible to save or reassemble a collection using `make_collection()`
- Automatic addition of new texts to the collection (you need to add new facts as `fact_<number>.txt` to the [`/text`](https://github.com/vsha96/search_system/tree/main/text) directory and restart the collection assembly using `make_collection()`)
- The processed collection is saved and loaded as a pickle object (in the [`/obj`](https://github.com/vsha96/search_system/tree/main/obj) directory)



