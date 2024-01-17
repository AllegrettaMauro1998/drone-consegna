import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

def random_forest():

    '''
    Metodo di classificazione
    -------------------
    Classifica il dataset per la fase di training e testing
    '''

    # Caricamento dataset
    dataset = pd.read_csv('C:/Users/pc/Desktop/Progetto_iCon_23-24/dataset/stazione_molfetta_anno_2023.csv', sep=';', header=0)
    
    dataset.fillna(0, inplace=True)

    # Rimuove eventuali colonne non necessarie
    X = dataset.drop(['temperatura', 'precipitazione'], axis=1)

    print(dataset)

    # Assegna direttamente la variabile y ai valori binari (0 o 1)
    y = dataset['temperatura', 'precipitazione'].astype(int)

    
    non_numeric_columns = X.select_dtypes(exclude=['number']).columns

    X_train = X.select_dtypes(include=['number'])

    # Divide il dataset in set di addestramento e set di test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Creazione del modello di classificazione (Random Forest)
    model = RandomForestClassifier()

    # Addestramento del modello
    model.fit(X_train, y_train)

    # Effettua previsioni sul set di test
    y_pred = model.predict(X_test)

    # Valutazione delle prestazioni del modello
    accuracy = accuracy_score(y_test, y_pred)

    # Visualizzazione di un report di classificazione
    print('Classification Report:')
    print(classification_report(y_test, y_pred))

    # Salva il modello con joblib
    joblib.dump(model, 'modello_random_forest.sav')

    # Ora il modello puo' essere utilizzato per fare previsioni sui nuovi dati
    nuovi_dati = pd.DataFrame({
        'precipitazione': [0.00, 3.00, 6.20], 
        'temperatura': [0.00, 15.00, 45.00],
    })

    # Effettua previsioni sui nuovi dati
    nuove_previsioni = model.predict(nuovi_dati)

    # Aggiunge le previsioni al DataFrame dei nuovi dati
    nuovi_dati['predictions'] = nuove_previsioni

    # Salva il DataFrame con le nuove previsioni
    nuovi_dati.to_csv("dataset/dataset_edit.csv", index=False)