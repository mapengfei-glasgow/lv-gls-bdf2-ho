"""Graph-network surrogate of the dynamic LV model (lvsolver GLS P1/P1, Sec. 4.6).

  params.py         varied parameters: nominal values, ranges, unit-cube map
  sample_params.py  Sobol design (geometry x 8 parameters) -> params.csv
  generate.py       batch driver: rule-based fibres -> lvsolver.run --save-fields
  data.py           dataset of solver runs, graph and features
  graph.py          Dalton-style augmented graph (virtual node layers)
  meshgraphnet.py   Encode-Process-Decode network
  train.py          one-step + pushforward training, rollout validation
  evaluate.py       rollout on held-out geometries: u error, cavity volume, EF
"""
