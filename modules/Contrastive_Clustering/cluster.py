import json
import os
import torch
from torch.utils.data import Dataset, DataLoader

from modules.Contrastive_Clustering.All import ModifiedNetwork

model_save_path = "./modules/Contrastive_Clustering/save/model_checkpoint_6_300.tar"
dataset_path = "./modules/Contrastive_Clustering/test"
output_file_path = './data/community_data.json'
output_file_mult_path = './data/community_data_mult.json'
probabilities_file_path = './data/cluster_probabilities.json'


class FeatureVectorDataset(Dataset):
    def __init__(self, directory):
        super(FeatureVectorDataset, self).__init__()
        self.directory = directory
        self.files = os.listdir(directory)
        self.identifiers = []
        self.features = []
        self.load_features()

    def load_features(self):
        for file_name in self.files:
            file_path = os.path.join(self.directory, file_name)
            with open(file_path, 'r') as file:
                for line in file:
                    parts = line.strip().split()
                    identifier = parts[0]
                    features = [float(part) for part in parts[1:]]
                    self.identifiers.append(identifier)
                    self.features.append(features)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.identifiers[idx], torch.tensor(self.features[idx], dtype=torch.float32)


class ClusterPredictor:
    def __init__(self, model_save_path=model_save_path, dataset_path=dataset_path, output_file_path=output_file_path,
                 input_dim=20, feature_dim=20, class_num=50):
        self.model_save_path = model_save_path
        self.dataset_path = dataset_path
        self.output_file_path = output_file_path
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.class_num = class_num

        self.model = ModifiedNetwork(self.input_dim, self.feature_dim, self.class_num)
        self.load_model()

    def load_model(self):
        checkpoint = torch.load(self.model_save_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def predict(self):
        dataset = FeatureVectorDataset(self.dataset_path)
        loader = DataLoader(dataset, batch_size=128, shuffle=False)
        all_identifiers = []
        all_predictions = []
        all_probabilities = []  # For saving cluster probabilities
        top3_indices_list = []  # To store top 3 indices for new functionality
        self.model.eval()
        with torch.no_grad():
            for identifiers, features in loader:
                features = features.to(torch.device('cpu'))
                _, probabilities = self.model(features)
                predicted_clusters = torch.argmax(probabilities, dim=1)
                top3_values, top3_indices = torch.topk(probabilities, k=3, dim=1)
                all_identifiers.extend(identifiers)
                all_predictions.extend(predicted_clusters.tolist())
                all_probabilities.extend(probabilities.tolist())
                top3_indices_list.extend(top3_indices.tolist())
        return all_identifiers, all_predictions, all_probabilities, top3_indices_list

    def save_probabilities_to_json(self, identifiers, probabilities):
        data = [{"id": identifier, "probabilities": prob} for identifier, prob in zip(identifiers, probabilities)]
        with open(probabilities_file_path, 'w') as f:
            json.dump(data, f, indent=4)

    def save_to_json(self, identifiers, predicted_clusters):
        unique_clusters = sorted(set(predicted_clusters))
        cluster_mapping = {cluster: i + 1 for i, cluster in enumerate(unique_clusters)}
        mapped_clusters = [cluster_mapping[cluster] for cluster in predicted_clusters]

        nodes = [{"id": identifier, "group": mapped_cluster} for identifier, mapped_cluster in
                 zip(identifiers, mapped_clusters)]
        links = []
        cluster_to_identifiers = {}
        for identifier, mapped_cluster in zip(identifiers, mapped_clusters):
            if mapped_cluster not in cluster_to_identifiers:
                cluster_to_identifiers[mapped_cluster] = []
            cluster_to_identifiers[mapped_cluster].append(identifier)

        for cluster, ids in cluster_to_identifiers.items():
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    links.append({"source": ids[i], "target": ids[j], "value": 1})

        output_data = {"nodes": nodes, "links": links}
        with open(self.output_file_path, 'w') as f:
            json.dump(output_data, f, indent=4)

    def generate_graph_data_v2(self, identifiers, top3_indices):
        graph_data = {
            "GraphData": {
                "node": [],
                "links": [],
                "group": [],
                "subgroups": [],
                "subsubgroups": []
            }
        }

        # Initialize groups to hold nodes based on top3 indices
        groups = {i: [] for i in range(max(max(top3_indices)) + 1)}
        subgroups = {i: [] for i in range(max(max(top3_indices)) + 1)}
        subsubgroups = {i: [] for i in range(max(max(top3_indices)) + 1)}

        for identifier, indices in zip(identifiers, top3_indices):
            # Create node entry
            graph_data["GraphData"]["node"].append({"id": identifier, "propertyValue": 1})

            # Assign node to groups based on top3 indices
            for idx, group_index in enumerate(indices):
                if idx == 0:
                    groups.setdefault(group_index, []).append(identifier)
                elif idx == 1:
                    subgroups.setdefault(group_index, []).append(identifier)
                elif idx == 2:
                    subsubgroups.setdefault(group_index, []).append(identifier)

        # Convert groups to required format and remove empty groups
        graph_data["GraphData"]["group"] = [group for group in [list(set(g)) for g in groups.values()] if group]
        graph_data["GraphData"]["subgroups"] = [group for group in [list(set(g)) for g in subgroups.values()] if group]
        graph_data["GraphData"]["subsubgroups"] = [group for group in [list(set(g)) for g in subsubgroups.values()] if group]

        # Generate links within each primary group
        for group in groups.values():
            if len(group) > 1:
                first_node = group[0]
                for other_node in group[1:]:
                    graph_data["GraphData"]["links"].append({
                        "source": first_node,
                        "target": other_node,
                        "value": 1
                    })
        for group in subgroups.values():
            if len(group) > 1:
                first_node = group[0]
                for other_node in group[1:]:
                    graph_data["GraphData"]["links"].append({
                        "source": first_node,
                        "target": other_node,
                        "value": 1
                    })
        for group in subsubgroups.values():
            if len(group) > 1:
                first_node = group[0]
                for other_node in group[1:]:
                    graph_data["GraphData"]["links"].append({
                        "source": first_node,
                        "target": other_node,
                        "value": 1
                    })

        return graph_data

    def save_graph_data_to_json(self, graph_data):
        if not os.path.exists(os.path.dirname(output_file_mult_path)):
            os.makedirs(os.path.dirname(output_file_mult_path))

        with open(output_file_mult_path, 'w') as f:
            json.dump(graph_data, f, indent=4)

    def run(self):
        identifiers, predicted_clusters, probabilities, top3_indices = self.predict()
        self.save_to_json(identifiers, predicted_clusters)
        self.save_probabilities_to_json(identifiers, probabilities)
        graph_data = self.generate_graph_data_v2(identifiers, top3_indices)
        self.save_graph_data_to_json(graph_data)

# predictor = ClusterPredictor()
# predictor.run()
