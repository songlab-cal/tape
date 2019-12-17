def test_basic():
    import torch
    from tape import ProteinBertModel, ProteinBertConfig, TAPETokenizer  # type: ignore

    config = ProteinBertConfig(hidden_size=12, intermediate_size=12 * 4, num_hidden_layers=2)
    model = ProteinBertModel(config)
    tokenizer = TAPETokenizer(vocab='iupac')

    sequence = 'GCTVEDRCLIGMGAILLNGCVIGSGSLVAAGALITQ'
    token_ids = torch.tensor([tokenizer.encode(sequence)])
    output = model(token_ids)
    sequence_output = output[0]  # noqa
    pooled_output = output[1]  # noqa
