import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import scipy.stats as stats

def analyze_review_distribution(review_counts):
    """Analisa a distribuição dos reviews e identifica outliers"""
    
    print("📊 ANALISANDO DISTRIBUIÇÃO DE REVIEWS")
    print(f"Total de jogos: {len(review_counts)}")
    
    # Estatísticas básicas
    original_mean = np.mean(review_counts)
    original_std = np.std(review_counts)
    original_median = np.median(review_counts)
    
    print(f"\n📈 Estatísticas Originais:")
    print(f"Média: {original_mean:.2f}")
    print(f"Desvio Padrão: {original_std:.2f}")
    print(f"Mediana: {original_median:.2f}")
    print(f"Razão Desvio/Média: {original_std/original_mean:.2f}")
    
    # Aplicando transformação logarítmica (adicionando 1 para evitar log(0))
    log_reviews = np.log1p(review_counts)  # log(1 + x)
    
    log_mean = np.mean(log_reviews)
    log_std = np.std(log_reviews)
    
    print(f"\n📊 Estatísticas Logarítmicas:")
    print(f"Média (log): {log_mean:.2f}")
    print(f"Desvio Padrão (log): {log_std:.2f}")
    
    return log_reviews, log_mean, log_std

def identify_outliers_log_normal(review_counts, z_threshold=2.5):
    """Identifica outliers usando distribuição log-normal"""
    
    log_reviews, log_mean, log_std = analyze_review_distribution(review_counts)
    
    # Calculando limites na escala logarítmica
    lower_bound_log = log_mean - z_threshold * log_std
    upper_bound_log = log_mean + z_threshold * log_std
    
    # Convertendo de volta para escala original
    lower_bound = np.expm1(lower_bound_log)  # exp(x) - 1
    upper_bound = np.expm1(upper_bound_log)
    
    print(f"\n🎯 Limites de Outliers (Z-score = {z_threshold}):")
    print(f"Limite inferior: {lower_bound:.2f} reviews")
    print(f"Limite superior: {upper_bound:.2f} reviews")
    
    # Identificando outliers
    outliers_mask = (review_counts < lower_bound) | (review_counts > upper_bound)
    outliers_count = np.sum(outliers_mask)
    
    print(f"\n🚫 Outliers identificados: {outliers_count} jogos ({outliers_count/len(review_counts)*100:.2f}%)")
    
    return outliers_mask, lower_bound, upper_bound

def identify_outliers_iqr(review_counts, multiplier=1.5):
    """Identifica outliers usando IQR na escala logarítmica"""
    
    log_reviews = np.log1p(review_counts)
    
    Q1 = np.percentile(log_reviews, 25)
    Q3 = np.percentile(log_reviews, 75)
    IQR = Q3 - Q1
    
    lower_bound_log = Q1 - multiplier * IQR
    upper_bound_log = Q3 + multiplier * IQR
    
    # Convertendo para escala original
    lower_bound = np.expm1(lower_bound_log)
    upper_bound = np.expm1(upper_bound_log)
    
    print(f"\n📏 Limites IQR (Multiplicador = {multiplier}):")
    print(f"Q1 (log): {Q1:.2f}, Q3 (log): {Q3:.2f}, IQR (log): {IQR:.2f}")
    print(f"Limite inferior: {lower_bound:.2f} reviews")
    print(f"Limite superior: {upper_bound:.2f} reviews")
    
    outliers_mask = (review_counts < lower_bound) | (review_counts > upper_bound)
    outliers_count = np.sum(outliers_mask)
    
    print(f"🚫 Outliers IQR: {outliers_count} jogos ({outliers_count/len(review_counts)*100:.2f}%)")
    
    return outliers_mask, lower_bound, upper_bound

def identify_outliers_percentile(review_counts, lower_percentile=1, upper_percentile=99):
    """Identifica outliers usando percentis"""
    
    lower_bound = np.percentile(review_counts, lower_percentile)
    upper_bound = np.percentile(review_counts, upper_percentile)
    
    print(f"\n📊 Limites por Percentil ({lower_percentile}%-{upper_percentile}%):")
    print(f"Limite inferior: {lower_bound:.2f} reviews")
    print(f"Limite superior: {upper_bound:.2f} reviews")
    
    outliers_mask = (review_counts < lower_bound) | (review_counts > upper_bound)
    outliers_count = np.sum(outliers_mask)
    
    print(f"🚫 Outliers Percentil: {outliers_count} jogos ({outliers_count/len(review_counts)*100:.2f}%)")
    
    return outliers_mask, lower_bound, upper_bound

def plot_distributions(original_data, filtered_data, method_name):
    """Plota as distribuições antes e depois da filtragem"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot original
    ax1.hist(original_data, bins=50, alpha=0.7, color='red', edgecolor='black')
    ax1.set_title('Distribuição Original de Reviews')
    ax1.set_xlabel('Número de Reviews')
    ax1.set_ylabel('Frequência')
    ax1.grid(True, alpha=0.3)
    
    # Plot filtrado
    ax2.hist(filtered_data, bins=50, alpha=0.7, color='green', edgecolor='black')
    ax2.set_title(f'Distribuição Filtrada ({method_name})')
    ax2.set_xlabel('Número de Reviews')
    ax2.set_ylabel('Frequência')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'json_converted_dataset/distribution_comparison_{method_name}.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Carrega os dados
    input_file = Path('filtered_datasets/filtered_dataset.json')
    output_dir = Path('filtered_datasets')
    output_dir.mkdir(exist_ok=True)
    
    with open(input_file, 'r', encoding='utf-8') as f:
        games = json.load(f)
    
    # Extrai contagem de reviews
    review_counts = np.array([game.get('total_reviews', 0) for game in games.values()])
    appids = list(games.keys())
    
    print("🎮 INICIANDO ANÁLISE DE OUTLIERS")
    print("=" * 50)
    
    # Método 1: Distribuição Log-Normal com Z-score
    print("\n" + "="*50)
    print("MÉTODO 1: DISTRIBUIÇÃO LOG-NORMAL (Z-SCORE)")
    print("="*50)
    
    outliers_mask_1, lower_1, upper_1 = identify_outliers_log_normal(review_counts, z_threshold=2.5)
    
    # Método 2: IQR na escala logarítmica
    print("\n" + "="*50)
    print("MÉTODO 2: IQR LOGARÍTMICO")
    print("="*50)
    
    outliers_mask_2, lower_2, upper_2 = identify_outliers_iqr(review_counts, multiplier=1.5)
    
    # Método 3: Percentis
    print("\n" + "="*50)
    print("MÉTODO 3: PERCENTIS")
    print("="*50)
    
    outliers_mask_3, lower_3, upper_3 = identify_outliers_percentile(review_counts, lower_percentile=1, upper_percentile=99)
    
    # Pergunta ao usuário qual método usar
    print("\n" + "="*50)
    print("ESCOLHA DO MÉTODO DE FILTRAGEM")
    print("="*50)
    print("1. Distribuição Log-Normal (Z-score = 2.5)")
    print("2. IQR Logarítmico (1.5 × IQR)")
    print("3. Percentis (1%-99%)")
    print("4. Combinar métodos (mais conservador)")
    
    choice = input("\nEscolha o método (1-4): ").strip()
    
    if choice == "1":
        outliers_mask = outliers_mask_1
        method_name = "log_normal_zscore"
        print(f"✅ Usando método Log-Normal. Removendo {np.sum(outliers_mask)} outliers.")
    elif choice == "2":
        outliers_mask = outliers_mask_2
        method_name = "iqr_logarithmic"
        print(f"✅ Usando método IQR Logarítmico. Removendo {np.sum(outliers_mask)} outliers.")
    elif choice == "3":
        outliers_mask = outliers_mask_3
        method_name = "percentile"
        print(f"✅ Usando método Percentil. Removendo {np.sum(outliers_mask)} outliers.")
    elif choice == "4":
        # Combinação mais conservadora: considera outlier se for outlier em qualquer método
        outliers_mask = outliers_mask_1 | outliers_mask_2 | outliers_mask_3
        method_name = "combined_conservative"
        print(f"✅ Usando método Combinado. Removendo {np.sum(outliers_mask)} outliers.")
    else:
        print("❌ Escolha inválida. Usando método Log-Normal como padrão.")
        outliers_mask = outliers_mask_1
        method_name = "log_normal_zscore"
    
    # Filtra os dados
    filtered_games = {}
    outlier_games = {}
    
    for i, appid in enumerate(appids):
        if not outliers_mask[i]:
            filtered_games[appid] = games[appid]
        else:
            outlier_games[appid] = games[appid]
    
    # Estatísticas após filtragem
    filtered_reviews = np.array([game.get('total_reviews', 0) for game in filtered_games.values()])
    
    print(f"\n📊 RESULTADO DA FILTRAGEM:")
    print(f"Jogos originais: {len(games)}")
    print(f"Jogos após filtragem: {len(filtered_games)}")
    print(f"Jogos removidos: {len(outlier_games)}")
    print(f"Redução: {len(outlier_games)/len(games)*100:.2f}%")
    
    print(f"\n📈 ESTATÍSTICAS FILTRADAS:")
    print(f"Média: {np.mean(filtered_reviews):.2f}")
    print(f"Desvio Padrão: {np.std(filtered_reviews):.2f}")
    print(f"Mediana: {np.median(filtered_reviews):.2f}")
    print(f"Razão Desvio/Média: {np.std(filtered_reviews)/np.mean(filtered_reviews):.2f}")
    
    # Salva os datasets
    filtered_file = output_dir / f'filtered_dataset_{method_name}.json'
    outliers_file = output_dir / f'outliers_{method_name}.json'
    
    with open(filtered_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_games, f, indent=2, ensure_ascii=False)
    
    with open(outliers_file, 'w', encoding='utf-8') as f:
        json.dump(outlier_games, f, indent=2, ensure_ascii=False)
    
    # Gera gráficos
    plot_distributions(review_counts, filtered_reviews, method_name)
    
    # Salva relatório
    report = {
        'method_used': method_name,
        'original_count': len(games),
        'filtered_count': len(filtered_games),
        'outliers_count': len(outlier_games),
        'reduction_percentage': len(outlier_games)/len(games)*100,
        'original_stats': {
            'mean': float(np.mean(review_counts)),
            'std': float(np.std(review_counts)),
            'median': float(np.median(review_counts))
        },
        'filtered_stats': {
            'mean': float(np.mean(filtered_reviews)),
            'std': float(np.std(filtered_reviews)),
            'median': float(np.median(filtered_reviews))
        },
        'filtering_details': {
            'lower_bound': float(lower_1 if choice == "1" else lower_2 if choice == "2" else lower_3),
            'upper_bound': float(upper_1 if choice == "1" else upper_2 if choice == "2" else upper_3)
        }
    }
    
    with open(output_dir / f'filtering_report_{method_name}.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n💾 Arquivos salvos:")
    print(f"Dataset filtrado: {filtered_file}")
    print(f"Outliers: {outliers_file}")
    print(f"Relatório: {output_dir / f'filtering_report_{method_name}.json'}")
    print(f"Gráfico: {output_dir / f'distribution_comparison_{method_name}.png'}")

if __name__ == '__main__':
    main()