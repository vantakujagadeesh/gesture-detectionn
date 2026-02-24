import tensorflow as tf
from tensorflow.keras import layers, models

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Attention and Normalization
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x + res

def build_transformer_model(input_shape, output_shape, 
                          head_size=256, 
                          num_heads=4, 
                          ff_dim=4, 
                          num_transformer_blocks=4, 
                          mlp_units=[128], 
                          dropout=0.25, 
                          mlp_dropout=0.25):
    
    inputs = layers.Input(shape=input_shape)
    x = inputs
    
    # Transformer Blocks
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    # Global Average Pooling to flatten the sequence
    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    
    # MLP Head
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
        
    outputs = layers.Dense(output_shape, activation="softmax")(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    return model
