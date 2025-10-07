from pathlib import Path
import json
from datetime import datetime
from textwrap import dedent


def generate_classification_markdown(name: str, metrics: dict) -> str:
    per0 = metrics['per_class'].get('0', {})
    per1 = metrics['per_class'].get('1', {})
    return dedent(f"""
    ### {name}

    | Métrica | Classe 0 | Classe 1 | Macro F1 | Weighted F1 | Accuracy |
    |---------|---------:|---------:|---------:|------------:|---------:|
    | Precision | {per0.get('precision',0):.4f} | {per1.get('precision',0):.4f} | {metrics['macro_avg_f1']:.4f} | {metrics['weighted_avg_f1']:.4f} | {metrics['accuracy']:.4f} |
    | Recall | {per0.get('recall',0):.4f} | {per1.get('recall',0):.4f} |  |  |  |
    | F1 | {per0.get('f1',0):.4f} | {per1.get('f1',0):.4f} |  |  |  |
    | Support | {per0.get('support',0)} | {per1.get('support',0)} |  |  |  |
    """)


def generate_cluster_markdown(name: str, metrics: dict) -> str:
    sil = metrics.get('silhouette')
    db = metrics.get('davies_bouldin')
    ch = metrics.get('calinski_harabasz')
    return dedent(f"""
    ### {name} (Clustering)

    | Métrica | Valor |
    |---------|------:|
    | Silhouette | {sil if sil is not None else '-'} |
    | Davies-Bouldin | {db if db is not None else '-'} |
    | Calinski-Harabasz | {ch if ch is not None else '-'} |
    """)


def write_auto_metrics(output_path: Path, sections: list):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('# Métricas Automatizadas\n')
        f.write(f'_Gerado em: {datetime.utcnow().isoformat()}Z_\n\n')
        for s in sections:
            f.write(s.strip() + '\n\n')
