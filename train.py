import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input, GlobalAveragePooling1D, concatenate
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
import pandas as pd

file_path = r"you_data.xlsx"
df = pd.read_excel(file_path)
quantiles = df.quantile([0.2, 0.4, 0.6, 0.8])

def categorize(value, col):
    q1, q2, q3, q4 = quantiles[col]
    if value <= q1:
        return 0  
    elif value <= q2:
        return 1  
    elif value <= q3:
        return 2  
    elif value <= q4:
        return 3  
    else:
        return 4 

for col in ['MC ', 'GC ', 'PH ', 'T']:
    df[f'{col}_category'] = df[col].apply(lambda x: categorize(x, col))

df.head()

X = df[['MC ', 'GC ', 'PH ', 'T']].values  
y_mc = df['MC _category'].values
y_gc = df['GC _category'].values
y_ph = df['PH _category'].values
y_t = df['T_category'].values

X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
X = X.reshape((X.shape[0], 4, 1))  


y_mc = to_categorical(y_mc, num_classes=5)
y_gc = to_categorical(y_gc, num_classes=5)
y_ph = to_categorical(y_ph, num_classes=5)
y_t = to_categorical(y_t, num_classes=5)


X_train, X_test, y_mc_train, y_mc_test, y_gc_train, y_gc_test, y_ph_train, y_ph_test, y_t_train, y_t_test = train_test_split(
    X, y_mc, y_gc, y_ph, y_t, test_size=0.2, random_state=42)

X_train = np.expand_dims(X_train, axis=-1)  
X_test = np.expand_dims(X_test, axis=-1)    


input_layer = Input(shape=(4, 1))


conv1 = Conv1D(filters=128, kernel_size=2, activation='relu', padding='same')(input_layer)
pool1 = MaxPooling1D(pool_size=2)(conv1)  # 4 → 2

conv2 = Conv1D(filters=64, kernel_size=2, activation='relu', padding='same')(pool1)
pool2 = MaxPooling1D(pool_size=2)(conv2)  # 2 → 1

conv3 = Conv1D(filters=32, kernel_size=2, activation='relu', padding='same')(pool2)

flat = GlobalAveragePooling1D()(conv3)

mc_output = Dense(64, activation='relu')(flat)
mc_output = Dropout(0.5)(mc_output)
mc_output = Dense(5, activation='softmax', name='MC_output')(mc_output)

gc_output = Dense(64, activation='relu')(flat)
gc_output = Dropout(0.5)(gc_output)
gc_output = Dense(5, activation='softmax', name='GC_output')(gc_output)

ph_output = Dense(64, activation='relu')(flat)
ph_output = Dropout(0.5)(ph_output)
ph_output = Dense(5, activation='softmax', name='PH_output')(ph_output)

t_output = Dense(64, activation='relu')(flat)
t_output = Dropout(0.5)(t_output)
t_output = Dense(5, activation='softmax', name='T_output')(t_output)


model = Model(inputs=input_layer, outputs=[mc_output, gc_output, ph_output, t_output])


model.compile(optimizer='adam', 
              loss={'MC_output': 'categorical_crossentropy', 
                    'GC_output': 'categorical_crossentropy', 
                    'PH_output': 'categorical_crossentropy', 
                    'T_output': 'categorical_crossentropy'}, 
              metrics={'MC_output': 'accuracy', 
                       'GC_output': 'accuracy', 
                       'PH_output': 'accuracy', 
                       'T_output': 'accuracy'})


history = model.fit(X_train, 
                    {'MC_output': y_mc_train, 
                     'GC_output': y_gc_train, 
                     'PH_output': y_ph_train, 
                     'T_output': y_t_train
                     }, 
                    epochs=30, 
                    batch_size=32, 
                    validation_data=(X_test, 
                                    {'MC_output': y_mc_test, 
                                     'GC_output': y_gc_test, 
                                     'PH_output': y_ph_test, 
                                     'T_output': y_t_test
                                     }))
