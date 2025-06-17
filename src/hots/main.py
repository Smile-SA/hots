import asyncio
from typing import Optional, Dict, Any
import logging
from pathlib import Path
import click

from core.models.instance import OptimizationInstance
from core.services.data_processor import DataProcessor
from core.services.placement_engine import PlacementEngine
from core.services.conflict_detector import ConflictDetector
from connectors.kafka_connector import KafkaConnector
from connectors.placement_connector import PlacementConnector
from data_structures import ApplicationConfig, LoopResults, SystemState

class MicroserviceOptimizer:
    """Orchestrateur principal pour l'optimisation de placement"""
    
    def __init__(self, config: ApplicationConfig):
        self.config = config
        self.logger = self._setup_logging()
        
        # Services
        self.data_processor = DataProcessor(config)
        self.placement_engine = PlacementEngine(config)
        self.conflict_detector = ConflictDetector(config)
        
        # Connectors
        self.kafka_connector = None
        self.placement_connector = PlacementConnector(config)
        
        # État
        self.current_state: Optional[SystemState] = None
        self.is_running = False
        self.loop_counter = 0
        
    def _setup_logging(self) -> logging.Logger:
        """Configure le logging"""
        logger = logging.getLogger('microservice_optimizer')
        handler = logging.FileHandler(
            Path(self.config.output_path) / 'optimizer.log'
        )
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger
    
    async def initialize(self) -> bool:
        """Initialise l'optimiseur"""
        try:
            self.logger.info("Initializing Microservice Optimizer...")
            
            # Initialiser les connecteurs si nécessaire
            if self.config.use_kafka:
                self.kafka_connector = KafkaConnector(self.config)
                await self.kafka_connector.connect()
            
            # Charger les données initiales
            await self.data_processor.load_initial_data()
            
            # Effectuer l'analyse initiale
            await self._perform_initial_analysis()
            
            self.logger.info("Initialization completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            return False
    
    async def _perform_initial_analysis(self):
        """Effectue l'analyse initiale (période T_init)"""
        self.logger.info("Starting initial analysis period...")
        
        # Récupérer les données initiales
        initial_data = await self.data_processor.get_initial_data()
        
        # Effectuer le clustering initial
        cluster_labels = await self.placement_engine.perform_initial_clustering(
            initial_data
        )
        
        # Calculer le placement initial
        initial_placement = await self.placement_engine.compute_initial_placement(
            initial_data, cluster_labels
        )
        
        # Créer l'état initial du système
        self.current_state = SystemState(
            timestamp=initial_data.timestamp,
            containers=initial_data.containers,
            cluster_profiles=initial_data.cluster_profiles,
            host_loads=initial_data.host_loads
        )
        
        self.logger.info("Initial analysis completed")
    
    async def start_optimization_loop(self):
        """Démarre la boucle d'optimisation principale"""
        self.logger.info("Starting optimization loop...")
        self.is_running = True
        
        try:
            while self.is_running:
                loop_start_time = asyncio.get_event_loop().time()
                
                # Incrémenter le compteur de boucle
                self.loop_counter += 1
                
                self.logger.info(f"Starting loop iteration {self.loop_counter}")
                
                # Récupérer les nouvelles données
                new_data = await self._get_next_data_batch()
                
                if not new_data:
                    self.logger.warning("No new data received, continuing...")
                    await asyncio.sleep(self.config.loop.tick)
                    continue
                
                # Traiter la nouvelle batch de données
                loop_results = await self._process_loop_iteration(new_data)
                
                # Sauvegarder les résultats
                await self._save_loop_results(loop_results)
                
                # Calculer le temps d'attente
                loop_duration = asyncio.get_event_loop().time() - loop_start_time
                wait_time = max(0, self.config.loop.tick - loop_duration)
                
                self.logger.info(
                    f"Loop {self.loop_counter} completed in {loop_duration:.2f}s, "
                    f"waiting {wait_time:.2f}s"
                )
                
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                
        except Exception as e:
            self.logger.error(f"Error in optimization loop: {e}")
        finally:
            await self._cleanup()
    
    async def _get_next_data_batch(self) -> Optional[SystemState]:
        """Récupère la prochaine batch de données"""
        if self.config.use_kafka and self.kafka_connector:
            return await self.kafka_connector.get_next_batch()
        else:
            return await self.data_processor.get_next_batch()
    
    async def _process_loop_iteration(self, new_data: SystemState) -> LoopResults:
        """Traite une itération de la boucle d'optimisation"""
        loop_start_time = asyncio.get_event_loop().time()
        
        results = LoopResults(
            loop_number=self.loop_counter,
            timestamp=new_data.timestamp
        )
        
        try:
            # Mettre à jour l'état du système
            self._update_system_state(new_data)
            
            # Détecter les conflits potentiels
            conflicts = await self.conflict_detector.detect_conflicts(
                self.current_state
            )
            
            if conflicts.has_clustering_conflicts():
                self.logger.info("Clustering conflicts detected, recomputing...")
                
                # Recalculer le clustering
                new_clustering = await self.placement_engine.update_clustering(
                    self.current_state, conflicts.clustering_conflicts
                )
                
                results.clustering_changes = len(new_clustering.changes)
                results.initial_silhouette = conflicts.clustering_metrics.silhouette_before
                results.final_silhouette = new_clustering.silhouette_after
                
            if conflicts.has_placement_conflicts():
                self.logger.info("Placement conflicts detected, recomputing...")
                
                # Recalculer le placement
                new_placement = await self.placement_engine.update_placement(
                    self.current_state, conflicts.placement_conflicts
                )
                
                results.placement_changes = len(new_placement.moves)
                
                # Exécuter les mouvements si nécessaire
                if new_placement.moves:
                    await self._execute_placement_moves(new_placement.moves)
            
            # Calculer les métriques de performance
            results.loop_duration = asyncio.get_event_loop().time() - loop_start_time
            results.objective_delta_after = await self._calculate_objective_delta()
            
        except Exception as e:
            self.logger.error(f"Error processing loop iteration: {e}")
            raise
        
        return results
    
    def _update_system_state(self, new_data: SystemState):
        """Met à jour l'état du système avec les nouvelles données"""
        if self.current_state is None:
            self.current_state = new_data
        else:
            # Fusionner les données en gardant une fenêtre glissante
            self.current_state = self.data_processor.merge_system_states(
                self.current_state, new_data
            )
    
    async def _execute_placement_moves(self, moves: list):
        """Exécute les mouvements de placement"""
        self.logger.info(f"Executing {len(moves)} placement moves...")
        
        for move in moves:
            try:
                success = await self.placement_connector.move_container(move)
                if success:
                    self.logger.info(f"Successfully moved {move.container_id}")
                else:
                    self.logger.warning(f"Failed to move {move.container_id}")
            except Exception as e:
                self.logger.error(f"Error moving container {move.container_id}: {e}")
    
    async def _calculate_objective_delta(self) -> float:
        """Calcule la fonction objectif actuelle"""
        return await self.placement_engine.calculate_objective(self.current_state)
    
    async def _save_loop_results(self, results: LoopResults):
        """Sauvegarde les résultats de la boucle"""
        await self.data_processor.save_loop_results(results)
    
    async def _cleanup(self):
        """Nettoie les ressources"""
        self.logger.info("Cleaning up resources...")
        
        if self.kafka_connector:
            await self.kafka_connector.disconnect()
        
        await self.placement_connector.close()
        await self.data_processor.cleanup()
    
    def stop(self):
        """Arrête l'optimiseur"""
        self.logger.info("Stopping optimizer...")
        self.is_running = False

# Interface CLI mise à jour
@click.command()
@click.argument('config_path', type=click.Path(exists=True))
@click.option('-k', '--clusters', type=int, help='Number of clusters')
@click.option('-t', '--tick', type=int, help='Loop tick duration (seconds)')
@click.option('--use-kafka', is_flag=True, help='Use Kafka for data streaming')
@click.option('-o', '--output', type=str, help='Output directory')
@click.option('--time-limit', type=int, help='Time limit in seconds')
async def main(config_path, clusters, tick, use_kafka, output, time_limit):
    """Point d'entrée principal de l'application"""
    
    # Charger la configuration
    config = ApplicationConfig.from_file(config_path)
    
    # Appliquer les overrides depuis la ligne de commande
    if clusters:
        config.clustering.nb_clusters = clusters
    if tick:
        config.loop.tick = tick
    if use_kafka:
        config.use_kafka = use_kafka
    if output:
        config.output_path = output
    if time_limit:
        config.optimization.time_limit = time_limit
    
    # Créer et démarrer l'optimiseur
    optimizer = MicroserviceOptimizer(config)
    
    try:
        # Initialiser
        if not await optimizer.initialize():
            click.echo("Failed to initialize optimizer", err=True)
            return 1
        
        # Démarrer la boucle d'optimisation
        await optimizer.start_optimization_loop()
        
    except KeyboardInterrupt:
        click.echo("Received interrupt signal, stopping...")
        optimizer.stop()
    except Exception as e:
        click.echo(f"Fatal error: {e}", err=True)
        return 1
    
    return 0

if __name__ == "__main__":
    asyncio.run(main())