from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from pathlib import Path
import yaml
import json
from enum import Enum

from core.models.instance import Container, Host, ClusterProfile, PlacementMove

class DataSourceType(Enum):
    """Types de sources de données"""
    KAFKA = "kafka"
    FILE = "file"
    API = "api"
    MOCK = "mock"

class ClusteringAlgorithm(Enum):
    """Algorithmes de clustering disponibles"""
    KMEANS = "kmeans"
    DBSCAN = "dbscan"
    HIERARCHICAL = "hierarchical"
    GAUSSIAN_MIXTURE = "gaussian_mixture"

class PlacementAlgorithm(Enum):
    """Algorithmes de placement disponibles"""
    FIRST_FIT = "first_fit"
    BEST_FIT = "best_fit"
    WORST_FIT = "worst_fit"
    GENETIC = "genetic"
    SIMULATED_ANNEALING = "simulated_annealing"

@dataclass
class KafkaConfig:
    """Configuration pour Kafka"""
    bootstrap_servers: List[str]
    topic_containers: str
    topic_hosts: str
    consumer_group: str
    auto_offset_reset: str = "latest"
    batch_size: int = 100
    timeout_ms: int = 5000
    security_protocol: str = "PLAINTEXT"
    sasl_mechanism: Optional[str] = None
    sasl_username: Optional[str] = None
    sasl_password: Optional[str] = None

@dataclass
class DataSourceConfig:
    """Configuration des sources de données"""
    type: DataSourceType
    kafka_config: Optional[KafkaConfig] = None
    file_paths: Dict[str, str] = field(default_factory=dict)
    api_endpoints: Dict[str, str] = field(default_factory=dict)
    refresh_interval_seconds: int = 30
    batch_size: int = 100
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DataSourceConfig':
        """Crée une configuration depuis un dictionnaire"""
        data_source_type = DataSourceType(data.get('type', 'file'))
        
        kafka_config = None
        if data_source_type == DataSourceType.KAFKA and 'kafka_config' in data:
            kafka_data = data['kafka_config']
            kafka_config = KafkaConfig(**kafka_data)
        
        return cls(
            type=data_source_type,
            kafka_config=kafka_config,
            file_paths=data.get('file_paths', {}),
            api_endpoints=data.get('api_endpoints', {}),
            refresh_interval_seconds=data.get('refresh_interval_seconds', 30),
            batch_size=data.get('batch_size', 100)
        )

@dataclass
class ClusteringConfig:
    """Configuration du clustering"""
    algorithm: ClusteringAlgorithm
    nb_clusters: int
    min_samples_per_cluster: int = 3
    max_clusters: int = 20
    silhouette_threshold: float = 0.5
    reclustering_threshold: float = 0.3
    feature_weights: Dict[str, float] = field(default_factory=lambda: {
        'cpu_usage': 0.3,
        'memory_usage': 0.3,
        'network_usage': 0.2,
        'storage_usage': 0.2
    })
    
    # Paramètres spécifiques aux algorithmes
    kmeans_params: Dict[str, Any] = field(default_factory=lambda: {
        'init': 'k-means++',
        'n_init': 10,
        'max_iter': 300,
        'random_state': 42
    })
    
    dbscan_params: Dict[str, Any] = field(default_factory=lambda: {
        'eps': 0.5,
        'min_samples': 5
    })
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ClusteringConfig':
        """Crée une configuration depuis un dictionnaire"""
        algorithm = ClusteringAlgorithm(data.get('algorithm', 'kmeans'))
        
        return cls(
            algorithm=algorithm,
            nb_clusters=data.get('nb_clusters', 5),
            min_samples_per_cluster=data.get('min_samples_per_cluster', 3),
            max_clusters=data.get('max_clusters', 20),
            silhouette_threshold=data.get('silhouette_threshold', 0.5),
            reclustering_threshold=data.get('reclustering_threshold', 0.3),
            feature_weights=data.get('feature_weights', {}),
            kmeans_params=data.get('kmeans_params', {}),
            dbscan_params=data.get('dbscan_params', {})
        )

@dataclass
class PlacementConfig:
    """Configuration du placement"""
    algorithm: PlacementAlgorithm
    max_moves_per_iteration: int = 10
    migration_cost_threshold: float = 0.5
    load_balancing_threshold: float = 0.8
    consolidation_threshold: float = 0.3
    
    # Poids pour la fonction objectif
    load_balancing_weight: float = 0.4
    resource_efficiency_weight: float = 0.3
    migration_cost_weight: float = 0.2
    affinity_satisfaction_weight: float = 0.1
    
    # Paramètres spécifiques aux algorithmes
    genetic_params: Dict[str, Any] = field(default_factory=lambda: {
        'population_size': 50,
        'generations': 100,
        'mutation_rate': 0.1,
        'crossover_rate': 0.8
    })
    
    simulated_annealing_params: Dict[str, Any] = field(default_factory=lambda: {
        'initial_temperature': 1000,
        'cooling_rate': 0.95,
        'min_temperature': 1,
        'max_iterations': 1000
    })
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PlacementConfig':
        """Crée une configuration depuis un dictionnaire"""
        algorithm = PlacementAlgorithm(data.get('algorithm', 'best_fit'))
        
        return cls(
            algorithm=algorithm,
            max_moves_per_iteration=data.get('max_moves_per_iteration', 10),
            migration_cost_threshold=data.get('migration_cost_threshold', 0.5),
            load_balancing_threshold=data.get('load_balancing_threshold', 0.8),
            consolidation_threshold=data.get('consolidation_threshold', 0.3),
            load_balancing_weight=data.get('load_balancing_weight', 0.4),
            resource_efficiency_weight=data.get('resource_efficiency_weight', 0.3),
            migration_cost_weight=data.get('migration_cost_weight', 0.2),
            affinity_satisfaction_weight=data.get('affinity_satisfaction_weight', 0.1),
            genetic_params=data.get('genetic_params', {}),
            simulated_annealing_params=data.get('simulated_annealing_params', {})
        )

@dataclass
class OptimizationConfig:
    """Configuration de l'optimisation"""
    time_limit: int = 300  # secondes
    memory_limit_mb: int = 1024
    parallel_threads: int = 4
    enable_preprocessing: bool = True
    enable_postprocessing: bool = True
    log_level: str = "INFO"
    save_intermediate_results: bool = False
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OptimizationConfig':
        """Crée une configuration depuis un dictionnaire"""
        return cls(
            time_limit=data.get('time_limit', 300),
            memory_limit_mb=data.get('memory_limit_mb', 1024),
            parallel_threads=data.get('parallel_threads', 4),
            enable_preprocessing=data.get('enable_preprocessing', True),
            enable_postprocessing=data.get('enable_postprocessing', True),
            log_level=data.get('log_level', "INFO"),
            save_intermediate_results=data.get('save_intermediate_results', False)
        )

@dataclass
class LoopConfig:
    """Configuration de la boucle d'optimisation"""
    tick: int = 60  # secondes entre chaque itération
    initial_analysis_duration: int = 300  # période T_init en secondes
    max_history_size: int = 1000  # nombre max d'échantillons gardés en mémoire
    sliding_window_size: int = 100  # taille de la fenêtre glissante
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LoopConfig':
        """Crée une configuration depuis un dictionnaire"""
        return cls(
            tick=data.get('tick', 60),
            initial_analysis_duration=data.get('initial_analysis_duration', 300),
            max_history_size=data.get('max_history_size', 1000),
            sliding_window_size=data.get('sliding_window_size', 100)
        )

@dataclass
class ApplicationConfig:
    """Configuration principale de l'application"""
    data_source: DataSourceConfig
    clustering: ClusteringConfig
    placement: PlacementConfig
    optimization: OptimizationConfig
    loop: LoopConfig
    output_path: str = "./results"
    use_kafka: bool = False
    debug_mode: bool = False
    
    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> 'ApplicationConfig':
        """Charge la configuration depuis un fichier YAML ou JSON"""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                data = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                data = json.load(f)
            else:
                raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
        
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ApplicationConfig':
        """Crée une configuration depuis un dictionnaire"""
        return cls(
            data_source=DataSourceConfig.from_dict(data.get('data_source', {})),
            clustering=ClusteringConfig.from_dict(data.get('clustering', {})),
            placement=PlacementConfig.from_dict(data.get('placement', {})),
            optimization=OptimizationConfig.from_dict(data.get('optimization', {})),
            loop=LoopConfig.from_dict(data.get('loop', {})),
            output_path=data.get('output_path', './results'),
            use_kafka=data.get('use_kafka', False),
            debug_mode=data.get('debug_mode', False)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit la configuration en dictionnaire"""
        return {
            'data_source': {
                'type': self.data_source.type.value,
                'file_paths': self.data_source.file_paths,
                'api_endpoints': self.data_source.api_endpoints,
                'refresh_interval_seconds': self.data_source.refresh_interval_seconds,
                'batch_size': self.data_source.batch_size
            },
            'clustering': {
                'algorithm': self.clustering.algorithm.value,
                'nb_clusters': self.clustering.nb_clusters,
                'min_samples_per_cluster': self.clustering.min_samples_per_cluster,
                'max_clusters': self.clustering.max_clusters,
                'silhouette_threshold': self.clustering.silhouette_threshold,
                'reclustering_threshold': self.clustering.reclustering_threshold,
                'feature_weights': self.clustering.feature_weights
            },
            'placement': {
                'algorithm': self.placement.algorithm.value,
                'max_moves_per_iteration': self.placement.max_moves_per_iteration,
                'migration_cost_threshold': self.placement.migration_cost_threshold,
                'load_balancing_threshold': self.placement.load_balancing_threshold
            },
            'optimization': {
                'time_limit': self.optimization.time_limit,
                'memory_limit_mb': self.optimization.memory_limit_mb,
                'parallel_threads': self.optimization.parallel_threads,
                'log_level': self.optimization.log_level
            },
            'loop': {
                'tick': self.loop.tick,
                'initial_analysis_duration': self.loop.initial_analysis_duration,
                'max_history_size': self.loop.max_history_size,
                'sliding_window_size': self.loop.sliding_window_size
            },
            'output_path': self.output_path,
            'use_kafka': self.use_kafka,
            'debug_mode': self.debug_mode
        }
    
    def save_to_file(self, config_path: Union[str, Path]):
        """Sauvegarde la configuration dans un fichier"""
        config_path = Path(config_path)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
            elif config_path.suffix.lower() == '.json':
                json.dump(self.to_dict(), f, indent=2)
            else:
                raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")

@dataclass
class SystemState:
    """État du système à un instant donné"""
    timestamp: datetime
    containers: Dict[str, Container]
    hosts: Dict[str, Host] = field(default_factory=dict)
    cluster_profiles: List[ClusterProfile] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_total_containers(self) -> int:
        """Retourne le nombre total de conteneurs"""
        return len(self.containers)
    
    def get_active_containers(self) -> List[Container]:
        """Retourne la liste des conteneurs actifs"""
        return [c for c in self.containers.values() if c.status.value == "running"]
    
    def get_host_utilization_stats(self) -> Dict[str, float]:
        """Retourne les statistiques d'utilisation des hôtes"""
        if not self.hosts:
            return {}
        
        utilizations = [host.get_utilization_score() for host in self.hosts.values()]
        
        return {
            'min_utilization': min(utilizations),
            'max_utilization': max(utilizations),
            'avg_utilization': sum(utilizations) / len(utilizations),
            'std_utilization': (sum((u - sum(utilizations)/len(utilizations))**2 for u in utilizations) / len(utilizations))**0.5
        }

@dataclass
class ConflictDetectionResult:
    """Résultat de la détection de conflits"""
    timestamp: datetime
    has_conflicts: bool
    clustering_conflicts: List[str] = field(default_factory=list)
    placement_conflicts: List[str] = field(default_factory=list)
    clustering_metrics: Optional[Dict[str, float]] = None
    placement_metrics: Optional[Dict[str, float]] = None
    
    def has_clustering_conflicts(self) -> bool:
        """Vérifie s'il y a des conflits de clustering"""
        return len(self.clustering_conflicts) > 0
    
    def has_placement_conflicts(self) -> bool:
        """Vérifie s'il y a des conflits de placement"""
        return len(self.placement_conflicts) > 0

@dataclass
class ClusteringResult:
    """Résultat d'une opération de clustering"""
    timestamp: datetime
    cluster_labels: List[int]
    cluster_profiles: List[ClusterProfile]
    silhouette_score: float
    inertia: Optional[float] = None
    changes: List[str] = field(default_factory=list)
    silhouette_after: float = 0.0
    
    def get_clusters_summary(self) -> Dict[int, int]:
        """Retourne un résumé des clusters (cluster_id -> nombre de conteneurs)"""
        summary = {}
        for label in self.cluster_labels:
            summary[label] = summary.get(label, 0) + 1
        return summary

@dataclass
class PlacementResult:
    """Résultat d'une opération de placement"""
    timestamp: datetime
    moves: List[PlacementMove]
    objective_value: float
    load_balance_score: float
    resource_efficiency_score: float
    total_migration_cost: float
    execution_time_seconds: float
    
    def get_moves_summary(self) -> Dict[str, int]:
        """Retourne un résumé des mouvements par hôte source"""
        summary = {}
        for move in self.moves:
            summary[move.from_host] = summary.get(move.from_host, 0) + 1
        return summary

@dataclass
class LoopResults:
    """Résultats d'une itération de la boucle d'optimisation"""
    loop_number: int
    timestamp: datetime
    clustering_changes: int = 0
    placement_changes: int = 0
    initial_silhouette: float = 0.0
    final_silhouette: float = 0.0
    objective_delta_after: float = 0.0
    loop_duration: float = 0.0
    errors: List[str] = field(default_factory=list)
    
    def has_errors(self) -> bool:
        """Vérifie s'il y a eu des erreurs"""
        return len(self.errors) > 0
    
    def get_improvement_percentage(self) -> float:
        """Calcule le pourcentage d'amélioration de l'objectif"""
        if self.initial_silhouette == 0:
            return 0.0
        return ((self.final_silhouette - self.initial_silhouette) / self.initial_silhouette) * 100

@dataclass
class OptimizationReport:
    """Rapport complet d'une session d'optimisation"""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_loops: int = 0
    total_clustering_operations: int = 0
    total_placement_operations: int = 0
    total_container_moves: int = 0
    avg_loop_duration: float = 0.0
    best_objective_value: float = float('inf')
    worst_objective_value: float = float('-inf')
    loop_results: List[LoopResults] = field(default_factory=list)
    
    def add_loop_result(self, result: LoopResults):
        """Ajoute un résultat de boucle au rapport"""
        self.loop_results.append(result)
        self.total_loops += 1
        
        if result.clustering_changes > 0:
            self.total_clustering_operations += 1
        
        if result.placement_changes > 0:
            self.total_placement_operations += 1
            self.total_container_moves += result.placement_changes
        
        # Mise à jour des moyennes
        total_duration = sum(lr.loop_duration for lr in self.loop_results)
        self.avg_loop_duration = total_duration / len(self.loop_results)
        
        # Mise à jour des min/max objectifs
        if result.objective_delta_after < self.best_objective_value:
            self.best_objective_value = result.objective_delta_after
        if result.objective_delta_after > self.worst_objective_value:
            self.worst_objective_value = result.objective_delta_after
    
    def get_success_rate(self) -> float:
        """Calcule le taux de succès des opérations"""
        if not self.loop_results:
            return 0.0
        
        successful_loops = sum(1 for lr in self.loop_results if not lr.has_errors())
        return successful_loops / len(self.loop_results) * 100
    
    def finalize(self):
        """Finalise le rapport"""
        self.end_time = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit le rapport en dictionnaire pour sérialisation"""
        return {
            'session_id': self.session_id,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'total_loops': self.total_loops,
            'total_clustering_operations': self.total_clustering_operations,
            'total_placement_operations': self.total_placement_operations,
            'total_container_moves': self.total_container_moves,
            'avg_loop_duration': self.avg_loop_duration,
            'best_objective_value': self.best_objective_value,
            'worst_objective_value': self.worst_objective_value,
            'success_rate': self.get_success_rate(),
            'loop_results_count': len(self.loop_results)
        }