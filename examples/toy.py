import os
from pathlib import Path

import numpy

from cururu.disk import save, load
from cururu.persistence import DuplicateEntryException
from pjdata.data import Data
from pjdata.data_creation import read_arff
from pjdata.dataset import Dataset
from pjdata.fastdata import FastData
from pjml.config.list import split
from pjml.pipeline import Pipeline
from pjml.tool.base.seq import Seq
from pjml.tool.data.communication.cache import Cache
from pjml.tool.data.communication.report import Report
from pjml.tool.data.evaluation.metric import Metric
from pjml.tool.data.evaluation.split import Split
from pjml.tool.data.flow.Save import Save
from pjml.tool.data.flow.applyusing import ApplyUsing
from pjml.tool.data.flow.file import File
from pjml.tool.data.flow.source import Source
from pjml.tool.data.flow.store import Store
from pjml.tool.data.manipulation.keep import Keep
from pjml.tool.data.modeling.supervised.classifier.dt import DT
from pjml.tool.data.modeling.supervised.classifier.nb import NB
from pjml.tool.data.modeling.supervised.classifier.svmc import SVMC
from pjml.tool.data.processing.feature.binarize import Binarize
from pjml.tool.data.processing.instance.sampler.over.random import OverS
from pjml.tool.macro import evaluator
from pjml.tool.meta.mfe import MFE
from pjdata import data

# ML 1 ========================================================================
# # Armazenar dataset, sem depender do pacote pjml.
# from cururu.pickleserver import PickleServer
#
# try:
#     PickleServer().store(read_arff('iris.arff'))
# except DuplicateEntryException:
#     pass
from pjml.tool.meta.wrap import Wrap

pipe = Pipeline(
    File('iris.arff'),
    Binarize(),
    # Source('messedup-dataset'),
    Keep(evaluator(
        Cache(
            ApplyUsing(
                NB()
            ),
            Metric(function='accuracy')
        )
    )),
    # Store(name='messedup-dataset', fields=['X', 'y', 'S']),
    Report(" $S for dataset {dataset.name}.")
    # Report("{history.last.config['function']} $S for dataset {dataset.name}.")
    ,
    MFE(),
    # Report("metafeats: $Md"),
    # Report("metafeats vals: $M"),
    # Report("$Xd $Xt"),
    # Report("$Yd $Yt"),
    # Save('saved.arff', '/tmp/')
)

# save('/tmp/dump/pipe0', pipe)


# print('--------\n', pipe.serialized)
# # print('--------\n', pipe.wrapped.serialized)
# save('/tmp/cururu/pipe', pipe)
# #
# pipe = load('/tmp/pipe')
# print(pipe)

print(111111)
dout = pipe.apply()
# save('/tmp/cururu/pipea', pipe)
print(222222)
dout = pipe.use()
print(333333)

exit(0)


print()
print('Testing nominal data...')
o = numpy.array([['a', 1], ['b', 2], ['c', 3], ['d', 4], ['e', 5]])
print(o, '<- all')
print()
d = Data(Dataset('asd', 'dsa'), X=o, Y=numpy.array([o[:, 1]]).T)
print(Split(partitions=2, partition=0).apply(d).X, '<- split 0')
print(Split(partitions=2, partition=1).apply(d).X, '<- split 1')

exit(0)
Save('lixo.arff').apply(dout)


# ML 2 ========================================================================
pipe = Pipeline(
    File('iris.arff'),

    OverS(sampling_strategy='not minority'),

    ApplyUsing(NB('bernoulli')),
    Metric(function='accuracy'),
    # Report('Accuracy: $r {history}'),
    Report('Accuracy: $r'),

    ApplyUsing(DT(max_depth=2)),
    Metric(function='accuracy'),
    Report('Accuracy: $r'),

    ApplyUsing(SVMC(kernel='linear')),
    Metric(function='accuracy'),
    Report('Accuracy: $r'),
)
dataout = pipe.apply()
dataout2 = pipe.use()

# ML 3 ========================================================================
pipe = Pipeline(
    File('iris.arff'),

    Cache(MFE()),
    Report('\nMeta-features Names: $Md \nMeta-features Values: $M \n  {name}')
)
dataout = pipe.apply()

"""
Problemas filosoficos

obs. Containers sempre contêm referências a outros transformers (sejam leves ou 
pesados) em config.

obs. Um mesmo pipeline pode gerar diversos históricos. 
GA não pode confiar no histórico, pois as mutações podem fazer com que data 
seja alterado e mude o comportamento do pipeline (trocando transformations);
ou seja, o GA deve ocorrer sobre o pipeline, não sobre as trasformations;
melhor dizendo, sobre o transformer, não sobre transformations

1
Antes era LEVEZA E 4 NÍVEIS - COM DEFEITO NO USE
Transformation(transformer, op)
Transformer(name, path, config)
Component(config)
Component
pros: basicamente dicts de strings = sem referências = menos memória
cons: havia o Component para o mesmo conceito, mas materializado


Solução atual = 1 abaixo
Estratégia: comparar desempenho das três em tempo (com e sem cache) e memória.


1 gasta mais espaço e mantém referências

Apply(transformer) / Use(transformer, training_data) 
Transformer(config)     <-  equivale a component+transformer
Transformer             <-  atalho para CS



2 gasta menos espaço e não mantém referências, mas burocratiza um pouco... 
(otimização prematura?)

Apply(transformer.serialized) / Use(transformer.serialized, training_data.uuid) 
Transformer(config)     <-  equivale a component+transformer
Transformer             <-  atalho para CS

    obs. Transformation não precisa de Transformer dentro dele. Quem precisar 
    pode
            materializá-lo.



3 espaço zero e sem referências, mas sem histórico

aply: data.uuid = uuid(data.uuid + transformer.uuid)
use:  data.uuid = uuid(data.uuid + transformer.uuid + training_data.uuid) 
Transformer(config)     <-  equivale a component+transformer
Transformer             <-  atalho para CS


4 usuário decide entre 1, 2 e 3; configuração seria numa das seguintes formas:
    a. variável global HISTORY=full|text|zero
    b. arg no apply/use cascateado automaticamente  <-- preferida 1
    c. arg no Transformer.__init__                  <-- fracassa para automl
    d. monkey-patch pjdata.data com pjdata.fastdata <-- preferida 2



Monkey patch:
from pjdata.fastdata import FastData
from pjdata import data
data.Data = FastData

Com abalone3.arff e PickleServer-speed
full: 45s/1.5s 204M  (prov dump do Data está levando Transformers junto)
zero: 45s/1.5s 40M

Com abalone3.arff e PickleServer-space-blosc
full: Illegal instruction (core dumped) 
zero: 47s/1.5s 10M

Com abalone3.arff e PickleServer-space-mono
full: 46s/1.9s 48M 
zero: 46s/1.5s 10M

Transformer dump não guarda model porque só tem name,path,config no dict for 
json.
"""
