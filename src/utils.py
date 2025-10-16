import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import time
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, RidgeCV
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,root_mean_squared_error
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor


def dataset_summary(df):
    # --- Dimensions ---
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns\n")

    # --- Basic statistics ---
    print("Basic statistics per column:")
    desc = df.describe().T
    print(desc[['min', 'mean', 'max']])

    # --- Standard deviation ---
    desc['std'] = df.std()
    print("\nStandard deviation:")
    print(desc['std'])

    # --- Group columns ---
    coord_cols = [col for col in df.columns if col.startswith(('x_', 'y_'))]
    vel_cols = [col for col in df.columns if col.startswith(('v_x', 'v_y'))]

    # --- Boxplots (valeurs absolues en log, comme avant) ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Coordinates
    data_coord = [np.abs(df[c]) + 1e-8 for c in coord_cols]
    axes[0].boxplot(data_coord, tick_labels=coord_cols)
    axes[0].set_yscale('log')
    axes[0].set_title("Coordinates (x, y) - log scale (abs values)")
    axes[0].set_xlabel("Variables")
    axes[0].set_ylabel("Values (log scale)")

    # Velocities
    data_vel = [np.abs(df[c]) + 1e-8 for c in vel_cols]
    axes[1].boxplot(data_vel, tick_labels=vel_cols)
    axes[1].set_yscale('log')
    axes[1].set_title("Velocities (v_x, v_y) - log scale (abs values)")
    axes[1].set_xlabel("Variables")
    axes[1].set_ylabel("Values (log scale)")





    plt.tight_layout()
    plt.show()

    # --- Histogrammes simples (pas de log) ---
    def plot_raw_hist(df, cols, color, group_name):
        for col in cols:
            plt.figure(figsize=(6, 3))
            plt.hist(df[col], bins=50, color=color, edgecolor='black')
            plt.title(f"{col} - Raw distribution ({group_name})")
            plt.xlabel("Value")
            plt.ylabel("Frequency")
    
            # ➕ Afficher min et max
            min_val = df[col].min()
            max_val = df[col].max()
            plt.annotate(f"min: {min_val:.2e}", xy=(0.98, 0.95), xycoords='axes fraction',
                         ha='right', va='top', fontsize=9, color='red')
            plt.annotate(f"max: {max_val:.2e}", xy=(0.98, 0.88), xycoords='axes fraction',
                         ha='right', va='top', fontsize=9, color='red')
    
            plt.tight_layout()
            plt.show()


    # Coordinates
    plot_raw_hist(df, coord_cols, color='skyblue', group_name='Coordinates')

    # Velocities
    plot_raw_hist(df, vel_cols, color='lightcoral', group_name='Velocities')


def clean_data(df, traj_len=257, tol=1e-8):
    """
    Nettoie un dataset de trajectoires.
    
    Étapes :
    1. Supprime les trajectoires dont la première ligne est entièrement zéro.
    2. Supprime les collisions : tronque chaque trajectoire à la première ligne où toutes les features sont proches de zéro.
    
    Args:
        df (pd.DataFrame): Dataset à nettoyer.
        traj_len (int): Longueur d'une trajectoire.
        tol (float): Tolérance pour considérer qu'une valeur est "zéro" pour la détection de collisions.
    
    Returns:
        pd.DataFrame: Dataset nettoyé.
    """
    # 1️⃣ Retirer la colonne ID si présente
    df_no_id = df.iloc[:, :-1] if df.shape[1] > traj_len else df.copy()
    
    # 2️⃣ Vérifier les trajectoires qui commencent par zéro
    num_traj = len(df_no_id) // traj_len
    zero_traj = []
    
    for i in range(num_traj):
        start_idx = i * traj_len
        first_row = df_no_id.iloc[start_idx]
        if (first_row == 0).all():
            zero_traj.append(i)
    
    # 3️⃣ Supprimer les trajectoires commençant par zéro
    drop_indices = []
    for traj_id in zero_traj:
        start = traj_id * traj_len
        end = (traj_id + 1) * traj_len
        drop_indices.extend(range(start, end))
    
    df_cleaned = df.drop(drop_indices).reset_index(drop=True)
    
    # 4️⃣ Supprimer les collisions
    cleaned = []
    num_traj = len(df_cleaned) // traj_len
    
    for i in range(num_traj):
        start = i * traj_len
        end = (i + 1) * traj_len
        traj = df_cleaned.iloc[start:end]
        
        traj_features = traj.iloc[:, :-1]  # toutes les colonnes sauf la dernière
        zero_mask = (np.abs(traj_features.values) < tol).all(axis=1)
        
        if zero_mask.any():
            first_zero = zero_mask.argmax()
            traj = traj.iloc[:first_zero]  # tronquer jusqu'à la première ligne zéro
        
        cleaned.append(traj)
    
    cleaned_df = pd.concat(cleaned).reset_index(drop=True)
    
    return cleaned_df


def split_trajectories(df, 
                                train_size=0.6, 
                                validation_size=0.2, 
                                test_size=0.2, 
                                method="random", 
                                random_state=None):

    if random_state is not None:
        np.random.seed(random_state)

    # Identifier les indices de début de trajectoire
    traj_start_indices = df.index[df['t'] == 0].tolist()
    traj_start_indices.append(len(df))  # ajouter la fin pour le dernier bloc

    # Construire les slices de trajectoires
    traj_slices = [(traj_start_indices[i], traj_start_indices[i+1]) for i in range(len(traj_start_indices)-1)]
    
    if method == "random":
        np.random.shuffle(traj_slices)
    else:
        raise NotImplementedError(f"Méthode {method} non implémentée pour l'instant")

    n_traj = len(traj_slices)
    n_train = int(train_size * n_traj)
    n_val = int(validation_size * n_traj)
    n_test = int(test_size * n_traj)

    # Ajustement si reste
    n_remaining = n_traj - (n_train + n_val + n_test)
    n_train += n_remaining

    train_slices = traj_slices[:n_train]
    val_slices = traj_slices[n_train:n_train+n_val]
    test_slices = traj_slices[n_train+n_val:n_train+n_val+n_test]

    # Concaténer les DataFrames correspondants
    train_df = pd.concat([df.iloc[start:end] for start, end in train_slices])
    val_df = pd.concat([df.iloc[start:end] for start, end in val_slices])
    test_df = pd.concat([df.iloc[start:end] for start, end in test_slices])

    return train_df, val_df, test_df

def replicate_initial_position_by_block(df):
    coords = ["x_1", "y_1",
              "x_2", "y_2",
              "x_3", "y_3"]
    
    copy = df.copy()
    data = copy[coords].values
    t_values = copy["t"].values
    
    # Détecter le début des blocs
    block_starts = np.where(t_values == 0)[0]
    block_starts = np.append(block_starts, len(df))  # ajouter fin du dernier bloc
    
    # Répliquer la position initiale pour chaque bloc
    for i in range(len(block_starts) - 1):
        start, end = block_starts[i], block_starts[i + 1]
        data[start:end] = data[start]  # réplication vectorisée
        
    
    copy[coords] = data
    return copy

def get_n_trajectories(df, n):
    start_indices = df.index[df['t'] == 0].tolist()
    trajectory_blocks = []

    for i in range(min(n, len(start_indices))):
        start_idx = start_indices[i]
        end_idx = start_indices[i + 1] if i + 1 < len(start_indices) else None

        # Convertir en positions avec get_loc()
        start_pos = df.index.get_loc(start_idx)
        end_pos = df.index.get_loc(end_idx) if end_idx else None

        trajectory_blocks.append(df.iloc[start_pos:end_pos])

    return pd.concat(trajectory_blocks)

def plot_y_yhat(y_test,y_pred, plot_title = "plot"):
    labels = ['x_1','y_1','x_2','y_2','x_3','y_3']
    MAX = 500
    if len(y_test) > MAX:
        idx = np.random.choice(len(y_test),MAX, replace=False)
    else:
        idx = np.arange(len(y_test))
    plt.figure(figsize=(10,10))
    for i in range(6):
        x0 = np.min(y_test[idx,i])
        x1 = np.max(y_test[idx,i])
        plt.subplot(3,2,i+1)
        plt.scatter(y_test[idx,i],y_pred[idx,i])
        plt.xlabel('True '+labels[i])
        plt.ylabel('Predicted '+labels[i])
        plt.plot([x0,x1],[x0,x1],color='red')
        plt.axis('square')
    plt.show()

def add_three_body_features(df, masses=(1,1,1), G=1.0):
    df_new = df.copy()
    eps = 1e-8
    m1, m2, m3 = masses

    # Positions
    r1 = df[['x_1', 'y_1']].values
    r2 = df[['x_2', 'y_2']].values
    r3 = df[['x_3', 'y_3']].values

    # --- Distances ---
    r12 = np.linalg.norm(r1 - r2, axis=1)
    r13 = np.linalg.norm(r1 - r3, axis=1)
    r23 = np.linalg.norm(r2 - r3, axis=1)

    df_new['r_12'] = r12
    df_new['r_13'] = r13
    df_new['r_23'] = r23

    # Inverses
    df_new['inv_r_12'] = 1.0 / (r12 + eps)
    df_new['inv_r_13'] = 1.0 / (r13 + eps)
    df_new['inv_r_23'] = 1.0 / (r23 + eps)

    # Ratios
    df_new['r12_over_r13'] = r12 / (r13 + eps)
    df_new['r12_over_r23'] = r12 / (r23 + eps)
    df_new['r13_over_r23'] = r13 / (r23 + eps)

    # --- Aire du triangle ---
    df_new['triangle_area'] = 0.5 * np.abs(
        (r2[:,0]-r1[:,0])*(r3[:,1]-r1[:,1]) - (r3[:,0]-r1[:,0])*(r2[:,1]-r1[:,1])
    )

    # --- Angles internes du triangle ---
    def angle(a, b, c):
        # loi des cosinus
        cos_angle = (b**2 + c**2 - a**2)/(2*b*c + eps)
        return np.arccos(np.clip(cos_angle, -1, 1))

    df_new['angle_1'] = angle(r23, r12, r13)
    df_new['angle_2'] = angle(r13, r12, r23)
    df_new['angle_3'] = angle(r12, r13, r23)

    # --- Centre de masse et distances au centre de masse ---
    X_cm = (m1*r1[:,0] + m2*r2[:,0] + m3*r3[:,0]) / (m1+m2+m3)
    Y_cm = (m1*r1[:,1] + m2*r2[:,1] + m3*r3[:,1]) / (m1+m2+m3)

    df_new['d1_cm'] = np.linalg.norm(r1 - np.stack([X_cm, Y_cm], axis=1), axis=1)
    df_new['d2_cm'] = np.linalg.norm(r2 - np.stack([X_cm, Y_cm], axis=1), axis=1)
    df_new['d3_cm'] = np.linalg.norm(r3 - np.stack([X_cm, Y_cm], axis=1), axis=1)

    # --- Moment angulaire approximatif autour du centre de masse ---
    # Lz = sum_i m_i * ((x_i - X_cm)*y_i - (y_i - Y_cm)*x_i) ; en 2D simplifié
    Lz = m1*((r1[:,0]-X_cm)*r1[:,1] - (r1[:,1]-Y_cm)*r1[:,0]) + \
         m2*((r2[:,0]-X_cm)*r2[:,1] - (r2[:,1]-Y_cm)*r2[:,0]) + \
         m3*((r3[:,0]-X_cm)*r3[:,1] - (r3[:,1]-Y_cm)*r3[:,0])
    df_new['Lz'] = Lz

    return df_new
