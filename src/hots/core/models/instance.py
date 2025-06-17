from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime
import numpy as np
from enum import Enum

class ContainerStatus(Enum):
    """États possibles d'un conteneur"""
    RUNNING = "running"
    PENDING = "pending"
    TERMINATING = "terminating"
    FAILED = "failed"

class HostStatus(Enum):
    """États possibles d'un hôte"""
    AVAILABLE = "available"
    MAINTENANCE = "maintenance"
    OVERLOADED = "overloaded"
    UNREACHABLE = "unreachable"

@dataclass
class ResourceRequirements:
    """Besoins en ressources d'un conteneur"""
    cpu_cores: float
    memory_gb: float
    storage_gb: float = 0.0
    network_bandwidth_mbps: float = 0.0
    gpu_count: int = 0
    
    def __post_init__(self):
        """Validation des valeurs"""
        if self.cpu_cores < 0 or self.memory_gb < 0:
            raise ValueError("Resource requirements cannot be negative")

@dataclass
class ResourceMetrics:
    """Métriques de consommation réelle d'un conteneur"""
    timestamp: datetime
    cpu_usage_percent: float
    memory_usage_gb: float
    storage_usage_gb: float = 0.0
    network_in_mbps: float = 0.0
    network_out_mbps: float = 0.0
    iops: float = 0.0
    
    def __post_init__(self):
        """Validation des métriques"""
        if not (0 <= self.cpu_usage_percent <= 100):
            raise ValueError("CPU usage must be between 0 and 100")
        if self.memory_usage_gb < 0:
            raise ValueError("Memory usage cannot be negative")

@dataclass
class Container:
    """Représente un conteneur dans le système"""
    id: str
    service_name: str
    status: ContainerStatus
    requirements: ResourceRequirements
    current_host: Optional[str] = None
    metrics_history: List[ResourceMetrics] = field(default_factory=list)
    labels: Dict[str, str] = field(default_factory=dict)
    affinity_rules: List[str] = field(default_factory=list)
    anti_affinity_rules: List[str] = field(default_factory=list)
    
    def get_latest_metrics(self) -> Optional[ResourceMetrics]:
        """Retourne les dernières métriques disponibles"""
        return self.metrics_history[-1] if self.metrics_history else None
    
    def get_average_usage(self, window_minutes: int = 10) -> Optional[ResourceMetrics]:
        """Calcule la consommation moyenne sur une fenêtre de temps"""
        if not self.metrics_history:
            return None
        
        cutoff_time = datetime.now()
        cutoff_time = cutoff_time.replace(minute=cutoff_time.minute - window_minutes)
        
        recent_metrics = [
            m for m in self.metrics_history 
            if m.timestamp >= cutoff_time
        ]
        
        if not recent_metrics:
            return None
        
        avg_cpu = sum(m.cpu_usage_percent for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_usage_gb for m in recent_metrics) / len(recent_metrics)
        avg_storage = sum(m.storage_usage_gb for m in recent_metrics) / len(recent_metrics)
        avg_net_in = sum(m.network_in_mbps for m in recent_metrics) / len(recent_metrics)
        avg_net_out = sum(m.network_out_mbps for m in recent_metrics) / len(recent_metrics)
        avg_iops = sum(m.iops for m in recent_metrics) / len(recent_metrics)
        
        return ResourceMetrics(
            timestamp=recent_metrics[-1].timestamp,
            cpu_usage_percent=avg_cpu,
            memory_usage_gb=avg_memory,
            storage_usage_gb=avg_storage,
            network_in_mbps=avg_net_in,
            network_out_mbps=avg_net_out,
            iops=avg_iops
        )

@dataclass
class HostCapacity:
    """Capacités d'un hôte"""
    cpu_cores: float
    memory_gb: float
    storage_gb: float
    network_bandwidth_mbps: float
    gpu_count: int = 0
    
    def can_accommodate(self, requirements: ResourceRequirements) -> bool:
        """Vérifie si l'hôte peut accueillir les besoins donnés"""
        return (
            self.cpu_cores >= requirements.cpu_cores and
            self.memory_gb >= requirements.memory_gb and
            self.storage_gb >= requirements.storage_gb and
            self.network_bandwidth_mbps >= requirements.network_bandwidth_mbps and
            self.gpu_count >= requirements.gpu_count
        )

@dataclass
class HostLoad:
    """Charge actuelle d'un hôte"""
    timestamp: datetime
    cpu_usage_percent: float
    memory_usage_gb: float
    storage_usage_gb: float = 0.0
    network_usage_mbps: float = 0.0
    active_containers: int = 0
    
    def get_utilization_vector(self) -> np.ndarray:
        """Retourne un vecteur de caractéristiques pour le clustering"""
        return np.array([
            self.cpu_usage_percent,
            self.memory_usage_gb,
            self.storage_usage_gb,
            self.network_usage_mbps,
            self.active_containers
        ])

@dataclass
class Host:
    """Représente un hôte dans l'infrastructure"""
    id: str
    hostname: str
    status: HostStatus
    capacity: HostCapacity
    current_load: Optional[HostLoad] = None
    load_history: List[HostLoad] = field(default_factory=list)
    zone: str = "default"
    labels: Dict[str, str] = field(default_factory=dict)
    
    def get_available_resources(self) -> Optional[ResourceRequirements]:
        """Calcule les ressources disponibles sur l'hôte"""
        if not self.current_load:
            return ResourceRequirements(
                cpu_cores=self.capacity.cpu_cores,
                memory_gb=self.capacity.memory_gb,
                storage_gb=self.capacity.storage_gb,
                network_bandwidth_mbps=self.capacity.network_bandwidth_mbps,
                gpu_count=self.capacity.gpu_count
            )
        
        # Calcul basé sur la charge actuelle
        available_cpu = self.capacity.cpu_cores * (1 - self.current_load.cpu_usage_percent / 100)
        available_memory = self.capacity.memory_gb - self.current_load.memory_usage_gb
        available_storage = self.capacity.storage_gb - self.current_load.storage_usage_gb
        available_bandwidth = self.capacity.network_bandwidth_mbps - self.current_load.network_usage_mbps
        
        return ResourceRequirements(
            cpu_cores=max(0, available_cpu),
            memory_gb=max(0, available_memory),
            storage_gb=max(0, available_storage),
            network_bandwidth_mbps=max(0, available_bandwidth),
            gpu_count=self.capacity.gpu_count  # Simplifié pour les GPU
        )
    
    def get_utilization_score(self) -> float:
        """Calcule un score d'utilisation global de l'hôte"""
        if not self.current_load:
            return 0.0
        
        cpu_util = self.current_load.cpu_usage_percent / 100
        memory_util = self.current_load.memory_usage_gb / self.capacity.memory_gb
        storage_util = self.current_load.storage_usage_gb / self.capacity.storage_gb
        
        # Score pondéré
        return (cpu_util * 0.4 + memory_util * 0.4 + storage_util * 0.2)

@dataclass
class ClusterProfile:
    """Profil d'un cluster de conteneurs similaires"""
    cluster_id: int
    containers: List[str]  # IDs des conteneurs
    centroid: np.ndarray  # Centroïde du cluster
    avg_cpu_usage: float
    avg_memory_usage: float
    avg_network_usage: float = 0.0
    service_types: Set[str] = field(default_factory=set)
    
    def get_representative_requirements(self) -> ResourceRequirements:
        """Retourne les besoins représentatifs du cluster"""
        return ResourceRequirements(
            cpu_cores=self.avg_cpu_usage / 100,  # Conversion pourcentage -> cores
            memory_gb=self.avg_memory_usage,
            network_bandwidth_mbps=self.avg_network_usage
        )

@dataclass
class PlacementMove:
    """Représente un mouvement de conteneur"""
    container_id: str
    from_host: str
    to_host: str
    reason: str
    estimated_cost: float = 0.0
    priority: int = 0  # 0 = haute priorité
    
    def __lt__(self, other):
        """Pour le tri par priorité"""
        return self.priority < other.priority

@dataclass
class OptimizationObjective:
    """Fonction objectif pour l'optimisation"""
    load_balancing_weight: float = 0.4
    resource_efficiency_weight: float = 0.3
    migration_cost_weight: float = 0.2
    affinity_satisfaction_weight: float = 0.1
    
    def __post_init__(self):
        """Validation des poids"""
        total = (self.load_balancing_weight + self.resource_efficiency_weight + 
                self.migration_cost_weight + self.affinity_satisfaction_weight)
        if abs(total - 1.0) > 1e-6:
            raise ValueError("Objective weights must sum to 1.0")

@dataclass
class OptimizationInstance:
    """Instance complète d'un problème d'optimisation"""
    timestamp: datetime
    containers: Dict[str, Container]
    hosts: Dict[str, Host]
    cluster_profiles: List[ClusterProfile]
    objective: OptimizationObjective
    constraints: Dict[str, any] = field(default_factory=dict)
    
    def get_placement_matrix(self) -> np.ndarray:
        """Retourne la matrice de placement actuelle (conteneurs x hôtes)"""
        container_ids = list(self.containers.keys())
        host_ids = list(self.hosts.keys())
        
        matrix = np.zeros((len(container_ids), len(host_ids)))
        
        for i, container_id in enumerate(container_ids):
            container = self.containers[container_id]
            if container.current_host and container.current_host in host_ids:
                j = host_ids.index(container.current_host)
                matrix[i, j] = 1
        
        return matrix
    
    def get_host_loads_vector(self) -> np.ndarray:
        """Retourne le vecteur des charges des hôtes"""
        return np.array([
            host.get_utilization_score() 
            for host in self.hosts.values()
        ])
    
    def get_containers_by_cluster(self, cluster_id: int) -> List[Container]:
        """Retourne les conteneurs d'un cluster donné"""
        cluster_profile = next(
            (cp for cp in self.cluster_profiles if cp.cluster_id == cluster_id),
            None
        )
        
        if not cluster_profile:
            return []
        
        return [
            self.containers[cid] 
            for cid in cluster_profile.containers 
            if cid in self.containers
        ]
    
    def calculate_migration_cost(self, move: PlacementMove) -> float:
        """Calcule le coût de migration d'un mouvement"""
        container = self.containers.get(move.container_id)
        if not container:
            return float('inf')
        
        # Coût basé sur la taille des données à migrer
        latest_metrics = container.get_latest_metrics()
        if latest_metrics:
            # Coût proportionnel à l'utilisation mémoire et stockage
            base_cost = (latest_metrics.memory_usage_gb + 
                        latest_metrics.storage_usage_gb) * 0.1
        else:
            base_cost = (container.requirements.memory_gb + 
                        container.requirements.storage_gb) * 0.1
        
        # Facteur de distance (simplifié)
        from_host = self.hosts.get(move.from_host)
        to_host = self.hosts.get(move.to_host)
        
        if from_host and to_host and from_host.zone != to_host.zone:
            base_cost *= 2.0  # Coût plus élevé pour migration inter-zone
        
        return base_cost
    
    def validate_placement(self) -> List[str]:
        """Valide le placement actuel et retourne les violations"""
        violations = []
        
        # Vérifier les capacités des hôtes
        for host_id, host in self.hosts.items():
            containers_on_host = [
                c for c in self.containers.values() 
                if c.current_host == host_id
            ]
            
            total_cpu = sum(c.requirements.cpu_cores for c in containers_on_host)
            total_memory = sum(c.requirements.memory_gb for c in containers_on_host)
            
            if total_cpu > host.capacity.cpu_cores:
                violations.append(f"Host {host_id} CPU overcommitted: {total_cpu} > {host.capacity.cpu_cores}")
            
            if total_memory > host.capacity.memory_gb:
                violations.append(f"Host {host_id} Memory overcommitted: {total_memory} > {host.capacity.memory_gb}")
        
        # Vérifier les règles d'affinité
        for container in self.containers.values():
            if container.current_host:
                host = self.hosts[container.current_host]
                
                # Vérifier les anti-affinités
                for rule in container.anti_affinity_rules:
                    conflicting_containers = [
                        c for c in self.containers.values()
                        if (c.current_host == container.current_host and 
                            c.id != container.id and 
                            rule in c.labels.values())
                    ]
                    
                    if conflicting_containers:
                        violations.append(
                            f"Anti-affinity violation: {container.id} and "
                            f"{[c.id for c in conflicting_containers]} on same host"
                        )
        
        return violations