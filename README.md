# Auto Encoder Example

Ref to [Building Autoencoders in Keras](https://blog.keras.io/building-autoencoders-in-keras.html)

# The Core Code

```
# the auto encoder
ipt = Input(shape=(28 * 28, ))
encoded = Dense(32, activation='relu')(encoded)
decoded = Dense(28*28, activation='sigmoid')(decoded)

auto_encoder = Model(ipt, decoded)
auto_encoder.compile(optimizer='adadelta', loss='binary_crossentropy')

# select the encoder
encoder = Model(ipt, encoded)

# select the decooder
# make a new Input()
decoder_ipt = Input(shape=(32,))
decode_layer = auto_encoder.layers[-1]
decoder = Model(decoder_ipt, decode_layer(decoder_ipt))
```
