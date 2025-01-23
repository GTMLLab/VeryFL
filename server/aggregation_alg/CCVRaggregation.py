
from ..base.baseAggregator import ServerAggregator
import numpy as np


class CCVRAggregator(ServerAggregator):
    def __init__(self):
        """
        ��ʼ�� CCVR Aggregator
        """
        super().__init__()

    def _on_before_aggregation(self):
        """
        �ۺ�ǰ�Ĳ�������ѡ��
        """
        pass

    def _on_after_aggregation(self):
        """
        �ۺϺ�Ĳ�������ѡ��
        """
        pass

    def _aggregate_alg(self, client_statistics_list=None):
        """
        ʵ�� CCVR �ۺ��㷨
        - client_statistics_list: �ͻ����ϴ���ͳ����Ϣ�б�
            [
                {class_label: (mu, sigma, n_samples), ...},
                ...
            ]
        """
        if client_statistics_list is None:
            client_statistics_list = self.model_pool

        global_statistics = {}
        for client_stats in client_statistics_list:
            for class_label, (mu, sigma, n_samples) in client_stats.items():
                if class_label not in global_statistics:
                    global_statistics[class_label] = {
                        "mu_sum": np.zeros_like(mu),
                        "sigma_sum": np.zeros_like(sigma),
                        "n_samples": 0
                    }

                global_statistics[class_label]["mu_sum"] += mu * n_samples
                global_statistics[class_label]["sigma_sum"] += sigma * n_samples
                global_statistics[class_label]["n_samples"] += n_samples

        # ����ȫ�־�ֵ��Э����
        aggregated_statistics = {}
        for class_label, stats in global_statistics.items():
            n_total = stats["n_samples"]
            if n_total > 0:
                mu_global = stats["mu_sum"] / n_total
                sigma_global = stats["sigma_sum"] / n_total
                aggregated_statistics[class_label] = (mu_global, sigma_global)

        return aggregated_statistics


if __name__ == '__main__':
    # ʾ�������������յ��Ŀͻ���ͳ����Ϣ
    client_statistics_list = [
        {
            0: (np.array([mu_0_c1]), np.array([sigma_0_c1]), n_0_c1),
            1: (np.array([mu_1_c1]), np.array([sigma_1_c1]), n_1_c1),
        },
        {
            0: (np.array([mu_0_c2]), np.array([sigma_0_c2]), n_0_c2),
            1: (np.array([mu_1_c2]), np.array([sigma_1_c2]), n_1_c2),
        },
    ]

    aggregator = CCVRAggregator()
    aggregated_statistics = aggregator._aggregate_alg(client_statistics_list)

    # ���ʾ��
    for class_label, (mu, sigma) in aggregated_statistics.items():
        print(f"Class {class_label}:")
        print(f"  Global Mu: {mu}")
        print(f"  Global Sigma: {sigma}")
