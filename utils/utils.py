import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import tensorflow as tf
from sklearn.svm import SVR
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_squared_log_error, root_mean_squared_error


#objetos usados para criar novos atriburos
coef_mod = {
              0: { 'DP-QPSK': 4,
                      'DP-16QAM': 8,
                      'DP-64QAM': 12
               },
               1: { 'DP-QPSK': 1/4,
                    'DP-16QAM': 1/8,
                    'DP-64QAM': 1/12
               }

}

osnr_label = r'$\mathrm{OSNR}_{\mathrm{NL}}$'
nlin_label = r'$\mathrm{NLIN}_{\mathrm{Power}}$'

out_params = {
	"PChOptdBm": "Optimal channel power [dBm]",
	"NLIN_Power": "Nonlinear interference noise",
	"OSNRdB_NL": "optical signal-to-noise ratio with NLIN interference",
	"BER_NL": "Bit error rate with NLIN interference",
	"OSNRdB": "Optical signal-to-noise ratio",
	"BER": "Bit error rate"
}

#Features
input_features = ['lightspeed', 'lambda', 'NumCh', 'ChSpacing', 'BaudRate', 'NumSpans', 'DispPar', 'Spans_gamma', 'Spans_beta2', 'Spans_alpha', 'Spans_L', 'Spans_PdBmCh',
                  'Spans_ModFormatCh', 'Spans_DeltaPdBIntLeft', 'Spans_ModFormatIntLeft', 'Spans_DeltaPdBIntRight', 'Spans_ModFormatIntRight', 'Spans_SpanLossdB'
]

string_features = ['Spans_ModFormatCh', 'Spans_ModFormatIntLeft', 'Spans_ModFormatIntRight']

const_input_features = ['lightspeed', 'lambda', 'ChSpacing', 'BaudRate', 'Spans_Fn']

zero_features = ['Spans_DeltaPdBIntLeft', 'Spans_DeltaPdBIntRight', 'Spans_SpanLossdB']

output_features = ['NLIN_Power', 'PChOptdBm',  'OSNRdB_NL', 'BER_NL', 'OSNRdB', 'BER']

coef_mod0 = ['coef_mod_Spans_ModFormatCh_0', 'coef_mod_Spans_ModFormatIntLeft_0', 'coef_mod_Spans_ModFormatIntRight_0']
coef_mod1 = ['coef_mod_Spans_ModFormatCh_1', 'coef_mod_Spans_ModFormatIntLeft_1', 'coef_mod_Spans_ModFormatIntRight_1']


parametros_cof1 = [
    {
    'NumCh': 5,
    'Spans_L': 80,
    'NumSpans': 15,
    'coef_mod_Spans_ModFormatCh_1': 1/4,
    'coef_mod_Spans_ModFormatIntLeft_1': 1/4,
    'coef_mod_Spans_ModFormatIntRight_1': 1/4,
    },
    {
    'Spans_PdBmCh': 2.5,
    'Spans_L': 80,
    'NumSpans': 15,
    'coef_mod_Spans_ModFormatCh_1': 1/8,
    'coef_mod_Spans_ModFormatIntLeft_1': 1/8,
    'coef_mod_Spans_ModFormatIntRight_1': 1/8,
    },
    {
    'NumCh': 5,
    'Spans_L': 80,
    'Spans_PdBmCh': 2.5,
    'coef_mod_Spans_ModFormatCh_1': 1/12,
    'coef_mod_Spans_ModFormatIntLeft_1': 1/12,
    'coef_mod_Spans_ModFormatIntRight_1': 1/12,
    }
]

params_coef1_1 = [
    {
    'coef_mod_Spans_ModFormatCh_1': 0.25,
    'coef_mod_Spans_ModFormatIntLeft_1': 0.25,
    'coef_mod_Spans_ModFormatIntRight_1': 0.25,
    'DispPar': 3.8,
    'Spans_gamma': 1.5,
    'Spans_alpha': 0.22
    },
    {
    'coef_mod_Spans_ModFormatCh_1': 0.25,
    'coef_mod_Spans_ModFormatIntLeft_1': 0.25,
    'coef_mod_Spans_ModFormatIntRight_1': 0.25,
    'DispPar': 16.7,
    'Spans_gamma': 1.3,
    'Spans_alpha': 0.2
    },
    {
    'coef_mod_Spans_ModFormatCh_1': 0.25,
    'coef_mod_Spans_ModFormatIntLeft_1': 0.25,
    'coef_mod_Spans_ModFormatIntRight_1': 0.25,
    'DispPar': 20.1,
    'Spans_gamma': 0.8,
    'Spans_alpha': 0.17
    }

]

color1 = ['b', 'g', 'r']
color2 = ['orange', 'c', 'y']
DisPar_to_Fiber = {
    20.1: "PSFC",
    16.7: "SMF",
    3.8: "NZDSF",
    0.5: "NZDSF",
    1.291411: "SMF",
    1.5: "PSFC"
}

params_coef0 = [
    {
    'coef_mod_Spans_ModFormatCh_0': 4,
    'coef_mod_Spans_ModFormatIntLeft_0': 4,
    'coef_mod_Spans_ModFormatIntRight_0': 4,
    'DispPar': 3.8,
    'Spans_gamma': 1.5,
    'Spans_alpha': 0.22
    },
    {
    'coef_mod_Spans_ModFormatCh_0': 4,
    'coef_mod_Spans_ModFormatIntLeft_0': 4,
    'coef_mod_Spans_ModFormatIntRight_0': 4,
    'DispPar': 16.7,
    'Spans_gamma': 1.3,
    'Spans_alpha': 0.2
    },
    {
    'coef_mod_Spans_ModFormatCh_0': 4,
    'coef_mod_Spans_ModFormatIntLeft_0': 4,
    'coef_mod_Spans_ModFormatIntRight_0': 4,
    'DispPar': 20.1,
    'Spans_gamma': 0.8,
    'Spans_alpha': 0.17
    }

]

# função utilizada para realizar a filtragem no dataset.
def filtrar(df, **kwargs):
  df_aux = df.copy()
  keys, values = list(kwargs.keys()), list(kwargs.values())
  filter = df_aux[keys[0]] == values[0]
  for key, value in zip(keys[1:], values[1:]):
    filter  = (filter) & (df_aux[key] == value)
  return df_aux[filter]



def model_evaluation(y_pred, y_test):
  return {
      "R²": round(r2_score(y_test, y_pred), 5),
      "MSE": round(mean_squared_error(y_test, y_pred), 5),
      "RMSE": round(root_mean_squared_error(y_test, y_pred), 5),
      "MAE": round(mean_absolute_error(y_test, y_pred), 5),

  }


def save_model(model, path):
  dump(model, path)

def load_model(path):
  return load(path)





def plot_scatter_comp(df, target, result_model_evaluation, figsize=(10,6), validation_mode = False):
  fig, ax = plt.subplots(figsize=figsize)
  label = {
    'OSNRdB_NL':  f'{osnr_label}',
    'NLIN_Power':  f'{nlin_label}'
  }[target]
  if validation_mode:
    title = f"Desempenho do modelo em relação\nao conjunto de validação:\nR² = {result_model_evaluation['R²']}; MAE = {result_model_evaluation['MAE']}\nMSE = {result_model_evaluation['MSE']},  RMSE = {round(result_model_evaluation['MSE']**0.5, 3)}"
  else:
    title= f"Estatísticas do modelo:\nR² = {result_model_evaluation['R²']}; MAE = {result_model_evaluation['MAE']}\nMSE = {result_model_evaluation['MSE']}; RMSE = {round(result_model_evaluation['MSE']**0.5, 3)}"
  
  target_pred = target + '_pred'
  ax.scatter(df[target], df[target_pred], label = 'Estimado')
  ax.plot(df[target], df[target], label=f'Curva ideal', color='r')
  if target == 'NLIN_power':
    ax.set(xlabel=f'{label} Real [dBm]', ylabel=f'{label} Estimado [dBm]')
  else:
     ax.set(xlabel=f'{label} Real [dB]', ylabel=f'{label} Estimado [dB]')
  ax.legend(
    alignment='left', loc='upper left', title=title)
  ax.grid()
  fig.tight_layout()
  fig.show()




def plot_all_fiber_pdBmCh(df_x, df_y, model, target, df_not_scaled=None, coef_mod=1/4, L=100, NCh=11, NSpam=15, scaled=False, figsize=(9,6)):
    label = {
    'OSNRdB_NL':  f'{osnr_label}',
    'NLIN_Power':  f'{nlin_label}',

  }[target]
    color_per_fiber = {
    'NZDSF': ['b', 'orange'],
    'SMF': ['g','c'],
    'PSFC': ['r', 'y']
    }
    target_pred = target + '_pred'

    df_test = df_x.copy()
    y_test = df_y.copy()

    y_validation2 = y_test.copy()
    y_validation2[target_pred] = model.predict(df_test)
    if scaled: 
        df_to_plot_predict = df_not_scaled.loc[df_test.index]
    else:
        df_to_plot_predict = df_test.copy()

    df_to_plot_predict[output_features + [target_pred]] = y_validation2

    fig, ax = plt.subplots(figsize=figsize)
    for DispPar in [3.8, 16.7, 20.1]:   
        df_filtrado = df_to_plot_predict[df_to_plot_predict['DispPar'] == DispPar].copy()
        df_to_plot = filtrar(df_filtrado, **{
            'NumCh': NCh,
            'Spans_L': L,
            'NumSpans': NSpam,
            'coef_mod_Spans_ModFormatCh_1': coef_mod,
            'coef_mod_Spans_ModFormatIntLeft_1': coef_mod,
            'coef_mod_Spans_ModFormatIntRight_1': coef_mod,
            })
        df_to_plot=df_to_plot.sort_values(by='Spans_PdBmCh')
        ax.plot(df_to_plot['Spans_PdBmCh'], df_to_plot[target], label=f'{label} (Real) - {DisPar_to_Fiber[DispPar]}', marker='o', color=color_per_fiber[DisPar_to_Fiber[DispPar]][0])
        ax.plot(df_to_plot['Spans_PdBmCh'], df_to_plot[target_pred], linestyle='--', label=f'{label} (Estimada) - {DisPar_to_Fiber[DispPar]}', marker='^', color=color_per_fiber[DisPar_to_Fiber[DispPar]][1])

    if target == 'NLIN_Power':
      ax.set(xlabel='Potência de entrada [dBm]', ylabel= f'{label} [dBm]')
    else:
      ax.set(xlabel='Potência de entrada [dBm]', ylabel= f'{label} [dB]')
    ax.legend(loc='upper left')
    ax.grid()
    plt.show()


def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [OSNRdB_NL]')
  plt.plot(hist['epoch'], hist['mae'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mae'],
           label = 'Val Error')
  plt.ylim([0,5])
  plt.legend()
  plt.grid(True)
  plt.show()

  print()
  print()

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [OSNRdB_NL]')
  plt.plot(hist['epoch'], hist['mse'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mse'],
           label = 'Val Error')
  plt.ylim([0,20])
  plt.legend()
  plt.grid(True)
  plt.show()


def cross_val_scores(build_model, X, y, feature, num_folds=5):
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=25)
    from sklearn.model_selection import KFold
    result = {
        'R²': [],
        'MAE': [],
        'MSE': [],
        'MEAN_MAE': None,
        'MEAN_MSE': None,
        'MEAN_R2': None
    }    
    kfold = KFold(n_splits=num_folds, shuffle=True)
    fold_n = 1
    for train_index, test_index in kfold.split(X, y):
        x_train, x_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        ann_model = build_model(x_train.shape)
        history = ann_model.fit(
            x_train,
            y_train,
            epochs=1000,
            batch_size=64,
            validation_split=0.3,
            callbacks=[early_stop]
        )
        metricas  = model_evaluation(ann_model.predict(x_test), y_test, x_train)
        print(f'Score for fold {fold_n}: {metricas}')
        result['MAE'].append(metricas['MAE'])
        result['MSE'].append(metricas['MSE'])
        result['R²'].append(metricas['R²'])
        

        fold_n += 1
    result.update({'MEAN_MAE': np.mean(result['MAE']),
                         'MEAN_MSE': np.mean(result['MSE']),
                         'MEAN_R2': np.mean(result['R²'])                 }
    )
    return result

def build_model(xtrain_shape):
    model = tf.keras.models.Sequential(
      [layers.Dense(64, activation='relu', input_shape=[xtrain_shape[1]]),
       layers.Dense(64, activation='sigmoid'),
       layers.Dense(1)
      ]
    )
    optimizer = tf.keras.optimizers.AdamW(learning_rate=0.001)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    return model


def crossval_scores(model, X, y, num_folds=5):
    from sklearn.model_selection import KFold
    result = {
        'R²': [],
        'MAE': [],
        'MSE': [],
        'MEAN_MAE': None,
        'MEAN_MSE': None,
        'MEAN_R2': None
    }    
    kfold = KFold(n_splits=num_folds, shuffle=True)
    fold_n = 1
    for train_index, test_index in kfold.split(X, y):
        x_train, x_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model.fit(x_train, y_train)
        metricas  = model_evaluation(model.predict(x_test), y_test, x_train)
        print(f'Score for fold {fold_n}: {metricas}')
        result['MAE'].append(metricas['MAE'])
        result['MSE'].append(metricas['MSE'])
        result['R²'].append(metricas['R²'])
        

        fold_n += 1
    result.update({'MEAN_MAE': np.mean(result['MAE']),
                         'MEAN_MSE': np.mean(result['MSE']),
                         'MEAN_R2': np.mean(result['R²'])}
    )
    return result