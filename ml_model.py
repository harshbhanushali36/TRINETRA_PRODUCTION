#!/usr/bin/env python3
"""
Scientifically Accurate Collision Prediction System - FIXED VERSION
Applied fixes:
- Increased screening distance to 200km for more conjunctions
- Enhanced ML training with synthetic data
- Robust handling for small datasets
- Proper physics-ML integration
"""
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm
import warnings
from scipy.spatial import cKDTree
from scipy.stats import norm
from sgp4.api import Satrec, jday
import requests
import os
import json
import gzip

warnings.filterwarnings('ignore')

# Physical Constants (IERS Standards)
MU_EARTH = 398600.4418  # km^3/s^2
EARTH_RADIUS = 6378.137  # km
J2 = 1.08262668e-3  # Earth's J2 coefficient
EARTH_ROTATION_RATE = 7.2921159e-5  # rad/s

@dataclass
class SpaceObject:
    """Space object with proper uncertainty modeling"""
    name: str
    norad_id: int
    state_vector: np.ndarray  # [x, y, z, vx, vy, vz] in km and km/s
    covariance_matrix: np.ndarray  # 6x6 covariance matrix
    epoch: datetime
    mass: float  # kg
    area: float  # m^2
    cd: float  # drag coefficient
    cr: float  # reflectivity coefficient
    object_type: str
    tle_line1: str
    tle_line2: str
    tle_age_days: float  # Age of TLE in days

class RealisticOrbitPropagator:
    """Orbit propagator with realistic uncertainty modeling"""
    
    def __init__(self):
        # Realistic atmospheric density model
        self.atmosphere_model = {
            200: 2.5e-10, 300: 1.9e-11, 400: 2.8e-12, 500: 6.9e-13,
            600: 2.1e-13, 700: 7.5e-14, 800: 3.0e-14, 1000: 5.6e-15, 1500: 2.0e-16
        }
        
    def create_anisotropic_covariance(self, obj: SpaceObject, pos: np.ndarray, vel: np.ndarray):
        """Create realistic anisotropic initial covariance based on orbit dynamics"""
        
        # Orbital elements for determining uncertainty directions
        r_mag = np.linalg.norm(pos)
        v_mag = np.linalg.norm(vel)
        
        # Radial, along-track, cross-track unit vectors
        r_unit = pos / r_mag
        h = np.cross(pos, vel)
        h_unit = h / np.linalg.norm(h)
        t_unit = np.cross(h_unit, r_unit)  # Along-track (tangential)
        
        # Base uncertainty from TLE age (different for each direction)
        if obj.tle_age_days < 1:
            sigma_radial = 0.5
            sigma_along_track = 1.0
            sigma_cross_track = 0.3
            sigma_radial_dot = 0.0005
            sigma_along_track_dot = 0.001
            sigma_cross_track_dot = 0.0003
        elif obj.tle_age_days < 7:
            sigma_radial = 1.0
            sigma_along_track = 5.0
            sigma_cross_track = 0.8
            sigma_radial_dot = 0.001
            sigma_along_track_dot = 0.005
            sigma_cross_track_dot = 0.0008
        else:
            sigma_radial = 2.0
            sigma_along_track = 20.0
            sigma_cross_track = 3.0
            sigma_radial_dot = 0.002
            sigma_along_track_dot = 0.015
            sigma_cross_track_dot = 0.002
        
        # Create rotation matrix from RSW to ECI
        rotation_matrix = np.column_stack([r_unit, t_unit, h_unit])
        
        # Covariance in RSW frame
        cov_rsw = np.zeros((6, 6))
        cov_rsw[0, 0] = sigma_radial**2
        cov_rsw[1, 1] = sigma_along_track**2
        cov_rsw[2, 2] = sigma_cross_track**2
        cov_rsw[3, 3] = sigma_radial_dot**2
        cov_rsw[4, 4] = sigma_along_track_dot**2
        cov_rsw[5, 5] = sigma_cross_track_dot**2
        
        # Position-velocity correlations
        cov_rsw[0, 3] = cov_rsw[3, 0] = 0.1 * sigma_radial * sigma_radial_dot
        cov_rsw[1, 4] = cov_rsw[4, 1] = 0.3 * sigma_along_track * sigma_along_track_dot
        cov_rsw[2, 5] = cov_rsw[5, 2] = 0.1 * sigma_cross_track * sigma_cross_track_dot
        
        # Transform to ECI frame
        rotation_6x6 = np.zeros((6, 6))
        rotation_6x6[:3, :3] = rotation_matrix
        rotation_6x6[3:, 3:] = rotation_matrix
        
        cov_eci = rotation_6x6 @ cov_rsw @ rotation_6x6.T
        
        return cov_eci
    
    def propagate_state_sgp4(self, obj: SpaceObject, target_epoch: datetime) -> Tuple[np.ndarray, np.ndarray]:
        """Propagate using SGP4 with realistic anisotropic covariance growth"""
        
        try:
            sat = Satrec.twoline2rv(obj.tle_line1, obj.tle_line2)
            jd, fr = jday(target_epoch.year, target_epoch.month, target_epoch.day,
                         target_epoch.hour, target_epoch.minute, target_epoch.second)
            
            e, r, v = sat.sgp4(jd, fr)
            
            if e != 0:
                return obj.state_vector.copy(), obj.covariance_matrix.copy()
            
            new_state = np.concatenate([np.array(r), np.array(v)])
            new_covariance = self.create_anisotropic_covariance(obj, np.array(r), np.array(v))
            
            # Additional uncertainty growth based on propagation time
            dt_hours = (target_epoch - obj.epoch).total_seconds() / 3600
            
            if abs(dt_hours) > 0:
                growth_factor = 1 + abs(dt_hours) / 24
                
                pos = new_state[:3]
                vel = new_state[3:]
                r_mag = np.linalg.norm(pos) if np.linalg.norm(pos) > 0 else 1.0
                
                r_unit = pos / r_mag
                h = np.cross(pos, vel)
                h_unit = h / np.linalg.norm(h) if np.linalg.norm(h) > 0 else np.array([0.,0.,1.])
                t_unit = np.cross(h_unit, r_unit)
                
                growth_radial = growth_factor
                growth_along_track = growth_factor**1.5
                growth_cross_track = growth_factor**0.8
                
                rotation_matrix = np.column_stack([r_unit, t_unit, h_unit])
                scaling_matrix = np.diag([growth_radial, growth_along_track, growth_cross_track])
                
                rotation_6x6 = np.zeros((6, 6))
                rotation_6x6[:3, :3] = rotation_matrix
                rotation_6x6[3:, 3:] = rotation_matrix
                
                scaling_6x6 = np.zeros((6, 6))
                scaling_6x6[:3, :3] = scaling_matrix
                scaling_6x6[3:, 3:] = scaling_matrix
                
                transform = rotation_6x6 @ scaling_6x6 @ rotation_6x6.T
                new_covariance = transform @ new_covariance @ transform.T
            
            return new_state, new_covariance
            
        except Exception as e:
            print(f"Propagation error: {e}")
            return obj.state_vector.copy(), obj.covariance_matrix.copy()

class ImprovedConjunctionAnalyzer:
    """Enhanced conjunction analyzer with encounter plane analysis"""
    
    def __init__(self, n_samples=50000):
        self.n_samples = n_samples
        self.propagator = RealisticOrbitPropagator()
    
    def find_time_of_closest_approach(self, obj1: SpaceObject, obj2: SpaceObject,
                                     screening_epoch: datetime, window_hours: float = 1) -> Optional[datetime]:
        """Find TCA using SGP4 propagation"""
        
        try:
            sat1 = Satrec.twoline2rv(obj1.tle_line1, obj1.tle_line2)
            sat2 = Satrec.twoline2rv(obj2.tle_line1, obj2.tle_line2)
            
            t_start = screening_epoch - timedelta(hours=window_hours)
            t_end = screening_epoch + timedelta(hours=window_hours)
            
            min_dist = float('inf')
            tca = screening_epoch
            
            # Coarse search
            current = t_start
            while current <= t_end:
                jd, fr = jday(current.year, current.month, current.day,
                            current.hour, current.minute, current.second)
                
                e1, r1, v1 = sat1.sgp4(jd, fr)
                e2, r2, v2 = sat2.sgp4(jd, fr)
                
                if e1 == 0 and e2 == 0:
                    dist = np.linalg.norm(np.array(r1) - np.array(r2))
                    if dist < min_dist:
                        min_dist = dist
                        tca = current
                
                current += timedelta(minutes=10)
            
            # Fine search
            t_start_fine = tca - timedelta(minutes=15)
            t_end_fine = tca + timedelta(minutes=15)
            
            current = t_start_fine
            while current <= t_end_fine:
                jd, fr = jday(current.year, current.month, current.day,
                            current.hour, current.minute, current.second)
                
                e1, r1, v1 = sat1.sgp4(jd, fr)
                e2, r2, v2 = sat2.sgp4(jd, fr)
                
                if e1 == 0 and e2 == 0:
                    dist = np.linalg.norm(np.array(r1) - np.array(r2))
                    if dist < min_dist:
                        min_dist = dist
                        tca = current
                
                current += timedelta(minutes=1)
            
            return tca
            
        except Exception as e:
            print(f"TCA search error: {e}")
            return screening_epoch
    
    def get_encounter_plane_transform(self, rel_pos: np.ndarray, rel_vel: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create encounter plane coordinate system"""
        
        rel_vel_mag = np.linalg.norm(rel_vel)
        if rel_vel_mag < 1e-6:
            z_axis = rel_pos / np.linalg.norm(rel_pos)
            x_axis = np.array([1, 0, 0])
            if abs(np.dot(z_axis, x_axis)) > 0.9:
                x_axis = np.array([0, 1, 0])
            x_axis = x_axis - np.dot(x_axis, z_axis) * z_axis
            x_axis = x_axis / np.linalg.norm(x_axis)
            y_axis = np.cross(z_axis, x_axis)
        else:
            z_axis = rel_vel / rel_vel_mag
            x_axis = rel_pos - np.dot(rel_pos, z_axis) * z_axis
            x_axis_mag = np.linalg.norm(x_axis)
            
            if x_axis_mag < 1e-6:
                if abs(z_axis[0]) < 0.9:
                    x_axis = np.array([1, 0, 0]) - z_axis[0] * z_axis
                else:
                    x_axis = np.array([0, 1, 0]) - z_axis[1] * z_axis
                x_axis = x_axis / np.linalg.norm(x_axis)
            else:
                x_axis = x_axis / x_axis_mag
            
            y_axis = np.cross(z_axis, x_axis)
        
        transform_matrix = np.array([x_axis, y_axis, z_axis])
        encounter_pos = transform_matrix @ rel_pos
        
        return transform_matrix, encounter_pos
    
    def project_covariance_to_encounter_plane(self, combined_cov: np.ndarray, transform_matrix: np.ndarray) -> np.ndarray:
        """Project 3D covariance to 2D encounter plane"""
        pos_cov_3d = combined_cov[:3, :3]
        encounter_cov_3d = transform_matrix @ pos_cov_3d @ transform_matrix.T
        encounter_cov_2d = encounter_cov_3d[:2, :2]
        return encounter_cov_2d
    
    def alfano_collision_probability(self, miss_distance_2d: np.ndarray, cov_2d: np.ndarray, 
                                   combined_radius_km: float) -> float:
        """Alfano (2005) method for 2D encounter plane collision probability"""
        
        try:
            det_cov = np.linalg.det(cov_2d)
            if det_cov < 1e-20 or np.any(np.isnan(cov_2d)):
                return 0.0
            
            miss_2d_mag = np.linalg.norm(miss_distance_2d)
            max_sigma = np.sqrt(np.max(np.linalg.eigvals(cov_2d)))
            if miss_2d_mag > 10 * max_sigma + combined_radius_km:
                return 0.0
            
            cov_inv = np.linalg.inv(cov_2d)
            mahal_dist_sq = miss_distance_2d.T @ cov_inv @ miss_distance_2d
            R_sq = combined_radius_km**2
            
            if mahal_dist_sq < 1e-10:
                pc = R_sq * np.sqrt(det_cov) / (2 * np.pi)
                pc = min(pc, 1.0)
            else:
                pc = (R_sq / (2 * np.pi)) * np.exp(-mahal_dist_sq / 2) / np.sqrt(det_cov)
            
            return min(pc, 1.0)
            
        except Exception as e:
            print(f"Alfano calculation error: {e}")
            return 0.0
    
    def importance_sampling_monte_carlo(self, miss_distance_2d: np.ndarray, cov_2d: np.ndarray,
                                      combined_radius_km: float, n_samples: int = None) -> float:
        """Improved Monte Carlo with importance sampling"""
        
        if n_samples is None:
            n_samples = self.n_samples
        
        try:
            miss_2d_mag = np.linalg.norm(miss_distance_2d)
            
            if miss_2d_mag > 3 * combined_radius_km:
                bias_factor = max(0.1, 1 - 0.6 * (combined_radius_km / miss_2d_mag))
                bias_center = miss_distance_2d * bias_factor
                importance_cov = cov_2d * 0.2

                samples = np.random.multivariate_normal(bias_center, importance_cov, n_samples)
                collisions_weighted = 0.0

                for sample in samples:
                    dist_to_miss = np.linalg.norm(sample - miss_distance_2d)
                    if dist_to_miss < combined_radius_km:
                        orig_density = self._multivariate_normal_pdf(sample, np.zeros(2), cov_2d)
                        imp_density = self._multivariate_normal_pdf(sample, bias_center, importance_cov)
                        if imp_density > 0 and orig_density > 0:
                            weight = orig_density / imp_density
                            collisions_weighted += weight

                pc = collisions_weighted / n_samples

                if pc == 0.0:
                    fallback_samples = max(200_000, n_samples * 4)
                    samples_fb = np.random.multivariate_normal(miss_distance_2d, cov_2d, fallback_samples)
                    distances = np.linalg.norm(samples_fb, axis=1)
                    pc = np.sum(distances < combined_radius_km) / fallback_samples

            else:
                samples = np.random.multivariate_normal(miss_distance_2d, cov_2d, n_samples)
                distances_to_origin = np.linalg.norm(samples, axis=1)
                collisions = np.sum(distances_to_origin < combined_radius_km)
                pc = collisions / n_samples
            
            return min(pc, 1.0)
            
        except Exception as e:
            print(f"Importance sampling MC error: {e}")
            return 0.0
    
    def _multivariate_normal_pdf(self, x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> float:
        """Compute multivariate normal PDF"""
        try:
            det_cov = np.linalg.det(cov)
            if det_cov < 1e-20:
                return 0.0
            
            diff = x - mean
            cov_inv = np.linalg.inv(cov)
            exponent = -0.5 * diff.T @ cov_inv @ diff
            
            return np.exp(exponent) / np.sqrt((2 * np.pi)**len(x) * det_cov)
        except:
            return 0.0
    
    def compute_collision_probability(self, rel_pos_km: np.ndarray, rel_vel_km: np.ndarray,
                                     combined_cov: np.ndarray, combined_radius_m: float) -> Tuple[float, float, Dict]:
        """Enhanced collision probability with encounter plane analysis"""
        
        miss_distance_km = np.linalg.norm(rel_pos_km)
        combined_radius_km = combined_radius_m / 1000.0
        
        rel_vel_estimate = rel_vel_km
        if np.linalg.norm(rel_vel_estimate) < 1e-6:
            r_mag = np.linalg.norm(rel_pos_km)
            orbital_speed = np.sqrt(MU_EARTH / r_mag) if r_mag > 0 else 7.5
            rel_vel_estimate = np.array([orbital_speed * 0.1, 0, 0])
        
        transform_matrix, encounter_pos = self.get_encounter_plane_transform(rel_pos_km, rel_vel_estimate)
        cov_2d = self.project_covariance_to_encounter_plane(combined_cov, transform_matrix)
        miss_distance_2d = encounter_pos[:2]
        
        if np.any(np.isnan(cov_2d)) or np.linalg.det(cov_2d) < 1e-20:
            return 0.0, 0.0, {}
        
        eigenvals = np.linalg.eigvals(cov_2d)
        max_sigma = float(np.sqrt(np.max(eigenvals)))
        min_sigma = float(np.sqrt(np.min(eigenvals)))
        
        if miss_distance_km > 10 * max_sigma + combined_radius_km:
            return 0.0, 0.0, {}
        
        mc_samples = max(self.n_samples, 50_000) if miss_distance_km > 2.0 else self.n_samples
        pc_mc = self.importance_sampling_monte_carlo(miss_distance_2d, cov_2d, combined_radius_km, mc_samples)
        pc_alfano = self.alfano_collision_probability(miss_distance_2d, cov_2d, combined_radius_km)
        
        diagnostics = {
            'encounter_plane_miss_distance_km': float(np.linalg.norm(miss_distance_2d)),
            'covariance_eigenvalues_km2': eigenvals.tolist(),
            'max_position_sigma_km': max_sigma,
            'min_position_sigma_km': min_sigma,
            'covariance_condition_number': (max_sigma / min_sigma) if min_sigma > 1e-10 else float('inf'),
            'monte_carlo_samples_used': int(mc_samples)
        }
        
        return float(pc_mc), float(pc_alfano), diagnostics

class MLRiskPredictor:
    """Machine Learning based risk prediction - FIXED VERSION"""
    
    def __init__(self):
        self.classifier = RandomForestClassifier(
            n_estimators=300, max_depth=20, min_samples_split=5,
            min_samples_leaf=2, max_features='sqrt', random_state=42, n_jobs=-1
        )
        self.regressor = RandomForestRegressor(
            n_estimators=300, max_depth=20, min_samples_split=5,
            min_samples_leaf=2, max_features='sqrt', random_state=42, n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.residual_std = 0
        self.residual_mean = 0
        
    def prepare_features(self, conjunction_data: dict) -> np.ndarray: 
        """Enhanced feature extraction"""
        features = []
        
        # Basic features
        miss_dist = conjunction_data.get('miss_distance_km', 0)
        rel_vel = conjunction_data.get('relative_velocity_ms', 0)
        combined_radius = conjunction_data.get('combined_radius_m', 0)
        pos_uncertainty = conjunction_data.get('position_uncertainty_km', 0)
        
        features.append(miss_dist)
        features.append(rel_vel)
        features.append(combined_radius)
        features.append(pos_uncertainty)
        
        # TLE ages
        tle_age1 = conjunction_data.get('tle_age_days_obj1', 0)
        tle_age2 = conjunction_data.get('tle_age_days_obj2', 0)
        features.extend([tle_age1, tle_age2, max(tle_age1, tle_age2), abs(tle_age1 - tle_age2)])
        
        # Probability features
        pc_mc = conjunction_data.get('pc_monte_carlo', 0)
        pc_alfano = conjunction_data.get('pc_alfano', 0)
        pc_max = conjunction_data.get('pc_maximum', 1e-15)
        features.extend([pc_mc, pc_alfano, np.log10(pc_max + 1e-15), abs(pc_mc - pc_alfano)])
        
        # Encounter plane
        encounter_miss = conjunction_data.get('encounter_plane_miss_distance_km', 0)
        max_sigma = conjunction_data.get('max_position_sigma_km', 0)
        min_sigma = conjunction_data.get('min_position_sigma_km', 0)
        features.extend([encounter_miss, max_sigma, min_sigma, max_sigma / max(min_sigma, 0.001)])
        
        # Covariance
        cond_num = conjunction_data.get('covariance_condition_number', 1)
        features.append(min(cond_num, 1000) if np.isfinite(cond_num) else 1000)
        
        # Object types
        type1 = conjunction_data.get('type1', 'SATELLITE')
        type2 = conjunction_data.get('type2', 'SATELLITE')
        features.extend([
            1 if type1 == 'DEBRIS' else 0, 1 if type2 == 'DEBRIS' else 0,
            1 if type1 == 'ROCKET_BODY' else 0, 1 if type2 == 'ROCKET_BODY' else 0,
            1 if 'DEBRIS' in [type1, type2] else 0, 1 if 'ROCKET_BODY' in [type1, type2] else 0
        ])
        
        type_risk = {'DEBRIS': 3, 'ROCKET_BODY': 2, 'SATELLITE': 1}
        features.append(type_risk.get(type1, 1) * type_risk.get(type2, 1))
        
        # Normalized features
        features.extend([
            miss_dist / max(pos_uncertainty, 0.1),
            miss_dist / max(max_sigma, 0.1),
            combined_radius / 1000 / max(miss_dist, 0.001)
        ])
        
        # Interactions
        features.extend([
            rel_vel * miss_dist, rel_vel * pos_uncertainty,
            np.log10(rel_vel + 1) * np.log10(miss_dist + 0.001)
        ])
        
        # Polynomial
        features.extend([
            miss_dist ** 2, np.sqrt(max(miss_dist, 0)),
            pos_uncertainty ** 2, 1 / max(miss_dist, 0.001)
        ])
        
        # Time features
        if 'tca' in conjunction_data and isinstance(conjunction_data['tca'], datetime):
            hours_to_tca = (conjunction_data['tca'] - datetime.now(timezone.utc)).total_seconds() / 3600
            features.extend([max(0, min(hours_to_tca, 168)), 1 if hours_to_tca < 24 else 0, 1 if hours_to_tca < 6 else 0])
        else:
            features.extend([24, 0, 0])
        
        # Statistical
        if max_sigma > 0 and min_sigma > 0:
            mahal_approx = miss_dist / np.sqrt(max_sigma * min_sigma)
            features.extend([mahal_approx, np.exp(-0.5 * mahal_approx**2)])
        else:
            features.extend([0, 0])
        
        # Risk bands
        features.extend([
            1 if miss_dist < 1 else 0, 1 if miss_dist < 5 else 0, 1 if miss_dist < 10 else 0,
            1 if pos_uncertainty < 1 else 0, 1 if pos_uncertainty < 5 else 0, 1 if pos_uncertainty > 10 else 0
        ])
        
        return np.array(features)
    
    def create_risk_categories(self, pc_values: np.ndarray) -> np.ndarray:
        """Create risk categories based on collision probability"""
        categories = np.zeros(len(pc_values), dtype=int)
        categories[pc_values > 1e-4] = 3  # EMERGENCY
        categories[(pc_values > 1e-5) & (pc_values <= 1e-4)] = 2  # HIGH
        categories[(pc_values > 1e-7) & (pc_values <= 1e-5)] = 1  # MEDIUM
        return categories
    
    def generate_synthetic_conjunction(self, risk_level: str) -> dict:
        """Generate realistic synthetic conjunction data"""
        
        if risk_level == 'EMERGENCY':
            miss_distance = np.random.uniform(0.05, 1.5)
            rel_velocity = np.random.normal(10000, 2000)
            pos_uncertainty = np.random.uniform(0.5, 2.0)
            pc_target = np.random.uniform(1e-4, 1e-3)
        elif risk_level == 'HIGH':
            miss_distance = np.random.uniform(1.0, 5.0)
            rel_velocity = np.random.normal(8000, 2500)
            pos_uncertainty = np.random.uniform(1.0, 4.0)
            pc_target = np.random.uniform(1e-5, 9e-5)
        elif risk_level == 'MEDIUM':
            miss_distance = np.random.uniform(3.0, 15.0)
            rel_velocity = np.random.normal(7000, 3000)
            pos_uncertainty = np.random.uniform(2.0, 8.0)
            pc_target = np.random.uniform(1e-7, 9e-6)
        else:  # LOW
            miss_distance = np.random.uniform(10.0, 50.0)
            rel_velocity = np.random.normal(5000, 3000)
            pos_uncertainty = np.random.uniform(5.0, 20.0)
            pc_target = np.random.uniform(1e-12, 9e-8)
        
        # Ensure realistic ranges
        miss_distance = max(0.05, miss_distance)
        rel_velocity = max(500, min(15000, abs(rel_velocity)))
        pos_uncertainty = max(0.1, pos_uncertainty)
        
        max_sigma = pos_uncertainty * np.random.uniform(2.0, 4.0)
        min_sigma = pos_uncertainty * np.random.uniform(0.3, 0.8)
        
        tle_age1 = max(0.1, min(30, np.random.exponential(3)))
        tle_age2 = max(0.1, min(30, np.random.exponential(3)))
        
        if risk_level in ['HIGH', 'EMERGENCY']:
            type_choices = ['DEBRIS', 'ROCKET_BODY', 'SATELLITE']
            type_probs = [0.5, 0.3, 0.2]
        else:
            type_choices = ['SATELLITE', 'DEBRIS', 'ROCKET_BODY']
            type_probs = [0.5, 0.3, 0.2]
        
        type1 = np.random.choice(type_choices, p=type_probs)
        type2 = np.random.choice(type_choices, p=type_probs)
        
        radius_map = {'DEBRIS': np.random.uniform(0.5, 2), 'ROCKET_BODY': np.random.uniform(3, 7), 'SATELLITE': np.random.uniform(5, 15)}
        combined_radius = radius_map[type1] + radius_map[type2]
        
        pc_monte_carlo = pc_target * np.random.lognormal(0, 0.15)
        pc_alfano = pc_target * np.random.lognormal(0, 0.1)
        encounter_miss = miss_distance * np.random.uniform(0.7, 0.95)
        
        eigenval1 = max_sigma**2
        eigenval2 = min_sigma**2
        
        hours_to_tca = np.random.exponential(12) if risk_level in ['HIGH', 'EMERGENCY'] else np.random.exponential(48)
        tca = datetime.now(timezone.utc) + timedelta(hours=hours_to_tca)
        
        return {
            'object1': f'SYNTH_{type1}_{np.random.randint(10000, 99999)}',
            'object2': f'SYNTH_{type2}_{np.random.randint(10000, 99999)}',
            'type1': type1, 'type2': type2, 'tca': tca,
            'miss_distance_km': float(miss_distance),
            'relative_velocity_ms': float(rel_velocity),
            'pc_monte_carlo': float(pc_monte_carlo),
            'pc_alfano': float(pc_alfano),
            'pc_maximum': float(max(pc_monte_carlo, pc_alfano)),
            'combined_radius_m': float(combined_radius),
            'position_uncertainty_km': float(pos_uncertainty),
            'tle_age_days_obj1': float(tle_age1),
            'tle_age_days_obj2': float(tle_age2),
            'encounter_plane_miss_distance_km': float(encounter_miss),
            'max_position_sigma_km': float(max_sigma),
            'min_position_sigma_km': float(min_sigma),
            'covariance_condition_number': float(max_sigma / max(min_sigma, 0.001)),
            'covariance_eigenvalues_km2': [float(eigenval1), float(eigenval2)],
            'monte_carlo_samples_used': 50000,
            'is_synthetic': True, 'synthetic_risk_level': risk_level
        }

    def train(self, real_data: pd.DataFrame, min_synthetic_per_class: int = 150):
        """Train with real data + balanced synthetic augmentation"""
        
        print("\n" + "="*70)
        print("TRAINING ML MODEL: REAL DATA + SYNTHETIC AUGMENTATION")
        print("="*70)
        
        print(f"\nReal data: {len(real_data)} samples")
        
        real_distribution = {0: 0, 1: 0, 2: 0, 3: 0}
        if len(real_data) > 0:
            real_pc = real_data['pc_maximum'].values
            real_categories = self.create_risk_categories(real_pc)
            unique, counts = np.unique(real_categories, return_counts=True)
            for cat, count in zip(unique, counts):
                real_distribution[cat] = count
            
            print("Real data distribution:")
            risk_names = ['LOW', 'MEDIUM', 'HIGH', 'EMERGENCY']
            for cat in range(4):
                if real_distribution[cat] > 0:
                    print(f"  {risk_names[cat]}: {real_distribution[cat]}")
        
        # Generate balanced synthetic data
        print(f"\nGenerating synthetic data ({min_synthetic_per_class} per class)...")
        synthetic_data = []
        risk_names = ['LOW', 'MEDIUM', 'HIGH', 'EMERGENCY']
        
        for cat_id, risk_name in enumerate(risk_names):
            current_count = real_distribution[cat_id]
            needed = max(min_synthetic_per_class - current_count, 50)
            
            print(f"  {risk_name}: Generating {needed} samples (have {current_count} real)")
            for _ in range(needed):
                synthetic_data.append(self.generate_synthetic_conjunction(risk_name))
        
        synthetic_df = pd.DataFrame(synthetic_data)
        print(f"\nGenerated {len(synthetic_df)} synthetic samples")
        
        # Combine datasets
        if len(real_data) > 0:
            training_data = pd.concat([real_data, synthetic_df], ignore_index=True)
            print(f"Total training data: {len(training_data)} ({len(real_data)} real + {len(synthetic_df)} synthetic)")
        else:
            training_data = synthetic_df
            print(f"Total training data: {len(training_data)} (all synthetic)")
        
        # Prepare features
        print("\nPreparing features...")
        X = []
        for _, row in training_data.iterrows():
            features = self.prepare_features(row.to_dict())
            X.append(features)
        
        X = np.array(X)
        y_prob = training_data['pc_maximum'].values
        y_cat = self.create_risk_categories(y_prob)
        
        # Final distribution
        unique, counts = np.unique(y_cat, return_counts=True)
        print(f"\nFinal training distribution:")
        risk_names = ['LOW', 'MEDIUM', 'HIGH', 'EMERGENCY']
        for cat, count in zip(unique, counts):
            print(f"  {risk_names[cat]}: {count}")
        
        # Split data
        test_size = 0.2 if len(X) >= 100 else 0.15
        
        try:
            X_train, X_test, y_cat_train, y_cat_test, y_prob_train, y_prob_test = train_test_split(
                X, y_cat, y_prob, test_size=test_size, random_state=42, stratify=y_cat
            )
        except ValueError:
            print("  Warning: Stratification failed, using random split")
            X_train, X_test, y_cat_train, y_cat_test, y_prob_train, y_prob_test = train_test_split(
                X, y_cat, y_prob, test_size=test_size, random_state=42
            )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train classifier
        print("\nTraining Random Forest Classifier...")
        self.classifier.fit(X_train_scaled, y_cat_train)
        
        # Train regressor
        print("Training Random Forest Regressor...")
        y_prob_train_log = np.log10(y_prob_train + 1e-15)
        self.regressor.fit(X_train_scaled, y_prob_train_log)
        
        # Store residual statistics
        y_pred_train = self.regressor.predict(X_train_scaled)
        residuals = y_prob_train_log - y_pred_train
        self.residual_std = np.std(residuals)
        self.residual_mean = np.mean(residuals)
        
        # Evaluate
        y_cat_pred = self.classifier.predict(X_test_scaled)
        accuracy = np.mean(y_cat_pred == y_cat_test)
        print(f"\nClassification Accuracy: {accuracy:.3f}")
        
        self.is_trained = True
        print("✓ ML model trained successfully!")

    def predict_risk(self, conjunction_data: dict) -> dict:
        """Return BOTH physics and ML predictions for comparison"""
        
        pc = conjunction_data.get('pc_maximum', 0)
        if pc > 1e-4:
            physics_category = 'EMERGENCY'
            physics_score = 3
        elif pc > 1e-5:
            physics_category = 'HIGH'
            physics_score = 2
        elif pc > 1e-7:
            physics_category = 'MEDIUM'
            physics_score = 1
        else:
            physics_category = 'LOW'
            physics_score = 0
        
        result = {
            'physics_risk_category': physics_category,
            'physics_risk_score': physics_score,
            'physics_collision_probability': pc,
        }
        
        if not self.is_trained:
            result.update({
                'ml_risk_category': 'NOT_TRAINED', 'ml_risk_score': -1,
                'ml_collision_probability': 0.0, 'ml_confidence': 0.0,
                'ml_uncertainty': 'N/A', 'ml_available': False
            })
            return result
        
        # ML prediction
        X = self.prepare_features(conjunction_data).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        # Classification
        risk_probabilities = self.classifier.predict_proba(X_scaled)[0]
        risk_category = np.argmax(risk_probabilities)
        
        # Regression with uncertainty
        all_tree_predictions = [tree.predict(X_scaled)[0] for tree in self.regressor.estimators_]
        collision_prob_log_mean = np.mean(all_tree_predictions)
        collision_prob_log_std = np.std(all_tree_predictions)
        
        total_uncertainty = np.sqrt(collision_prob_log_std**2 + self.residual_std**2)
        
        z_score = 1.96
        collision_prob_log_lower = collision_prob_log_mean - z_score * total_uncertainty
        collision_prob_log_upper = collision_prob_log_mean + z_score * total_uncertainty
        
        collision_prob = 10**collision_prob_log_mean
        collision_prob_lower = 10**collision_prob_log_lower
        collision_prob_upper = 10**collision_prob_log_upper
        
        # Entropy
        entropy = -np.sum(risk_probabilities * np.log(risk_probabilities + 1e-10))
        max_entropy = -np.log(1/len(risk_probabilities))
        normalized_entropy = entropy / max_entropy
        
        if normalized_entropy < 0.3 and collision_prob_log_std < 0.5:
            uncertainty_level = 'LOW'
        elif normalized_entropy < 0.6 and collision_prob_log_std < 1.0:
            uncertainty_level = 'MEDIUM'
        else:
            uncertainty_level = 'HIGH'
        
        risk_names = ['LOW', 'MEDIUM', 'HIGH', 'EMERGENCY']
        confidence = max(risk_probabilities) * 100 * (1 - normalized_entropy)
        
        result.update({
            'ml_risk_category': risk_names[risk_category],
            'ml_risk_score': int(risk_category),
            'ml_collision_probability': float(collision_prob),
            'ml_collision_probability_lower': float(collision_prob_lower),
            'ml_collision_probability_upper': float(collision_prob_upper),
            'ml_confidence': float(confidence),
            'ml_uncertainty': uncertainty_level,
            'ml_available': True,
            'category_agreement': physics_category == risk_names[risk_category],
            'probability_ratio': float(collision_prob / max(pc, 1e-15))
        })
        
        return result

    def save_model(self, filepath: str):
        """Save trained model"""
        if not self.is_trained:
            print("No trained model to save")
            return
        
        model_data = {
            'classifier': self.classifier, 'regressor': self.regressor,
            'scaler': self.scaler, 'is_trained': self.is_trained,
            'residual_std': self.residual_std, 'residual_mean': self.residual_mean
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(model_data, filepath)
        print(f"✓ Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> bool:
        """Load trained model"""
        if os.path.exists(filepath):
            model_data = joblib.load(filepath)
            self.classifier = model_data['classifier']
            self.regressor = model_data['regressor']
            self.scaler = model_data['scaler']
            self.is_trained = model_data['is_trained']
            self.residual_std = model_data.get('residual_std', 0)
            self.residual_mean = model_data.get('residual_mean', 0)
            print(f"✓ Model loaded from {filepath}")
            return True
        return False

class ScientificallyAccurateCollisionPredictor:
    """Main system with automated execution for GitHub workflow"""
    
    def __init__(self, monte_carlo_samples=50000):
        self.propagator = RealisticOrbitPropagator()
        self.analyzer = ImprovedConjunctionAnalyzer(n_samples=monte_carlo_samples)
        self.objects = []
        self.debug_log = []
        self.ml_predictor = MLRiskPredictor()
        self.training_data = []
        
    def create_object_from_tle(self, name: str, tle_line1: str, tle_line2: str,
                               epoch: datetime = None) -> Optional[SpaceObject]:
        """Create object with realistic anisotropic parameters"""
        
        try:
            satellite = Satrec.twoline2rv(tle_line1, tle_line2)
            
            if epoch is None:
                epoch = datetime.now(timezone.utc)
            
            jd, fr = jday(epoch.year, epoch.month, epoch.day,
                         epoch.hour, epoch.minute, epoch.second)
            
            e, r, v = satellite.sgp4(jd, fr)
            
            if e != 0:
                return None
            
            state_vector = np.concatenate([np.array(r), np.array(v)])
            
            # Calculate TLE age
            tle_epoch_year = 2000 + satellite.epochyr if satellite.epochyr < 57 else 1900 + satellite.epochyr
            tle_epoch = datetime(tle_epoch_year, 1, 1, tzinfo=timezone.utc) + timedelta(days=satellite.epochdays - 1)
            tle_age_days = (epoch - tle_epoch).total_seconds() / 86400
            
            # Object classification
            name_upper = name.upper()
            if 'DEB' in name_upper or 'FRAGMENT' in name_upper:
                obj_type = 'DEBRIS'
                mass = 10.0
                area = 0.1
                cd = 2.2
                cr = 1.5
            elif 'R/B' in name_upper or 'ROCKET' in name_upper:
                obj_type = 'ROCKET_BODY'
                mass = 1000.0
                area = 10.0
                cd = 2.2
                cr = 1.3
            else:
                obj_type = 'SATELLITE'
                mass = 500.0
                area = 5.0
                cd = 2.2
                cr = 1.5
            
            # Create anisotropic initial covariance
            covariance = self.propagator.create_anisotropic_covariance(
                SpaceObject(
                    name=name, norad_id=satellite.satnum, state_vector=state_vector,
                    covariance_matrix=np.eye(6), epoch=epoch, mass=mass, area=area,
                    cd=cd, cr=cr, object_type=obj_type, tle_line1=tle_line1,
                    tle_line2=tle_line2, tle_age_days=tle_age_days
                ),
                np.array(r), np.array(v)
            )
            
            return SpaceObject(
                name=name[:40], norad_id=satellite.satnum, state_vector=state_vector,
                covariance_matrix=covariance, epoch=epoch, mass=mass, area=area,
                cd=cd, cr=cr, object_type=obj_type, tle_line1=tle_line1,
                tle_line2=tle_line2, tle_age_days=tle_age_days
            )
            
        except Exception as e:
            print(f"Error creating object from TLE: {e}")
            return None
    
    def load_catalog(self, tle_data: List[Dict]) -> None:
        """Load catalog with validation"""
        
        print("\nLoading validated object catalog...")
        
        epoch = datetime.now(timezone.utc)
        
        for item in tqdm(tle_data, desc="Processing objects"):
            obj = self.create_object_from_tle(
                item['name'], item['tle_line1'], item['tle_line2'], epoch
            )
            
            if obj is not None:
                if not any(kw in obj.name.upper() for kw in ['ISS', 'TIANHE', 'ZARYA']):
                    self.objects.append(obj)
        
        print(f"Loaded {len(self.objects):,} validated objects")
        
        types = {}
        for obj in self.objects:
            types[obj.object_type] = types.get(obj.object_type, 0) + 1
        
        print("\nCatalog Statistics:")
        for obj_type, count in sorted(types.items()):
            print(f"   {obj_type}: {count:,}")
    
    def screen_conjunctions_accurate(self, time_window_hours: float = 24,
                                    screening_distance_km: float = 200) -> List[Dict]:  # CHANGED: 50km → 200km
        """Accurate conjunction screening with proper distance validation"""
        
        print(f"\nAccurate screening (distance < {screening_distance_km} km)")
        
        # Limit catalog size for performance
        max_objects = 2000
        if len(self.objects) > max_objects:
            print(f"   Sampling {max_objects} objects from {len(self.objects):,}")
            
            debris = [o for o in self.objects if o.object_type == 'DEBRIS']
            rockets = [o for o in self.objects if o.object_type == 'ROCKET_BODY']
            sats = [o for o in self.objects if o.object_type == 'SATELLITE']
            
            np.random.seed(42)
            
            selected = []
            selected.extend(debris[:min(600, len(debris))])
            selected.extend(rockets[:min(400, len(rockets))])
            selected.extend(sats[:min(1000, len(sats))])
            
            if len(selected) < max_objects and len(sats) > 1000:
                remaining = max_objects - len(selected)
                selected.extend(np.random.choice(sats[1000:], min(remaining, len(sats)-1000), replace=False))
            
            objects_to_screen = selected[:max_objects]
        else:
            objects_to_screen = self.objects
        
        print(f"   Screening {len(objects_to_screen):,} objects")
        
        conjunctions = []
        min_distance_found = float('inf')
        
        n_steps = min(12, int(time_window_hours / 2))
        time_steps = np.linspace(0, time_window_hours, n_steps)
        start_epoch = datetime.now(timezone.utc)
        
        for hours in tqdm(time_steps, desc="Time steps"):
            current_epoch = start_epoch + timedelta(hours=hours)
            
            positions = []
            valid_objects = []
            
            jd, fr = jday(current_epoch.year, current_epoch.month, current_epoch.day,
                         current_epoch.hour, current_epoch.minute, current_epoch.second)
            
            for obj in objects_to_screen:
                try:
                    sat = Satrec.twoline2rv(obj.tle_line1, obj.tle_line2)
                    e, r, v = sat.sgp4(jd, fr)
                    if e == 0:
                        positions.append(np.array(r))
                        valid_objects.append(obj)
                except:
                    continue
            
            if len(positions) < 2:
                continue
            
            positions = np.array(positions)
            tree = cKDTree(positions)
            pairs = tree.query_pairs(screening_distance_km, output_type='ndarray')
            
            for i, j in pairs:
                dist = np.linalg.norm(positions[i] - positions[j])
                min_distance_found = min(min_distance_found, dist)
                
                if dist <= screening_distance_km:
                    obj1, obj2 = valid_objects[i], valid_objects[j]
                    
                    if obj1.object_type == 'SATELLITE' and obj2.object_type == 'SATELLITE':
                        name1_base = obj1.name.split('-')[0] if '-' in obj1.name else obj1.name[:10]
                        name2_base = obj2.name.split('-')[0] if '-' in obj2.name else obj2.name[:10]
                        if name1_base == name2_base:
                            continue
                    
                    conjunctions.append({
                        'obj1': obj1, 'obj2': obj2,
                        'screening_epoch': current_epoch, 'screening_distance': dist
                    })
        
        if min_distance_found < float('inf'):
            print(f"DEBUG: Minimum distance found: {min_distance_found:.2f} km")
        else:
            print(f"DEBUG: No close approaches found")
        
        unique_conjunctions = {}
        for conj in conjunctions:
            key = tuple(sorted([conj['obj1'].name, conj['obj2'].name]))
            if key not in unique_conjunctions or conj['screening_distance'] < unique_conjunctions[key]['screening_distance']:
                unique_conjunctions[key] = conj
        
        final_conjunctions = sorted(unique_conjunctions.values(), key=lambda x: x['screening_distance'])
        
        if len(final_conjunctions) > 100:
            print(f"   Limiting to 100 closest approaches")
            final_conjunctions = final_conjunctions[:100]
        
        valid_conjunctions = []
        for conj in final_conjunctions:
            if conj['screening_distance'] <= screening_distance_km:
                valid_conjunctions.append(conj)
        
        print(f"Found {len(valid_conjunctions)} valid conjunctions within {screening_distance_km} km")
        
        return valid_conjunctions
    
    def analyze_conjunction(self, obj1: SpaceObject, obj2: SpaceObject,
                       screening_epoch: datetime) -> Optional[Dict]:
        """Analyze conjunction with PHYSICS ONLY - ML comes later"""
        
        tca = self.analyzer.find_time_of_closest_approach(obj1, obj2, screening_epoch)
        
        if tca is None:
            return None
        
        state1, cov1 = self.propagator.propagate_state_sgp4(obj1, tca)
        state2, cov2 = self.propagator.propagate_state_sgp4(obj2, tca)
        
        rel_pos = state1[:3] - state2[:3]
        rel_vel = state1[3:] - state2[3:]
        
        miss_distance = np.linalg.norm(rel_pos)
        rel_speed = np.linalg.norm(rel_vel)
        
        if miss_distance > 100:
            return None
        
        combined_cov = cov1 + cov2
        
        radius1 = {'DEBRIS': 1, 'ROCKET_BODY': 5, 'SATELLITE': 10}.get(obj1.object_type, 5)
        radius2 = {'DEBRIS': 1, 'ROCKET_BODY': 5, 'SATELLITE': 10}.get(obj2.object_type, 5)
        combined_radius = radius1 + radius2
        
        pc_mc, pc_alfano, diagnostics = self.analyzer.compute_collision_probability(
            rel_pos, rel_vel, combined_cov, combined_radius
        )
        
        if max(pc_mc, pc_alfano) < 1e-12:
            return None
        
        result = {
            'object1': obj1.name, 'object2': obj2.name,
            'type1': obj1.object_type, 'type2': obj2.object_type,
            'tca': tca, 'miss_distance_km': miss_distance,
            'relative_velocity_ms': rel_speed * 1000,
            'pc_monte_carlo': pc_mc, 'pc_alfano': pc_alfano,
            'pc_maximum': max(pc_mc, pc_alfano),
            'combined_radius_m': combined_radius,
            'position_uncertainty_km': np.sqrt(np.trace(combined_cov[:3, :3]) / 3),
            'tle_age_days_obj1': obj1.tle_age_days,
            'tle_age_days_obj2': obj2.tle_age_days
        }
        
        result.update(diagnostics)
        self.training_data.append(result)
        
        return result

    
    def run_analysis(self, time_window_hours: float = 24) -> pd.DataFrame:
        """Run complete enhanced analysis with BOTH physics and ML outputs"""
        
        print("\n" + "="*70)
        print("AUTOMATED COLLISION PREDICTION SYSTEM")
        print("Physics Analysis + ML Training & Prediction")
        print("="*70)
        
        self.debug_log = []
        self.training_data = []
        
        # PHASE 1: PHYSICS-BASED ANALYSIS
        print("\n" + "="*70)
        print("PHASE 1: PHYSICS-BASED CONJUNCTION ANALYSIS")
        print("="*70)
        
        potential_conjunctions = self.screen_conjunctions_accurate(time_window_hours)
        
        if not potential_conjunctions:
            print("No conjunctions found")
            return pd.DataFrame()
        
        print(f"\nAnalyzing {len(potential_conjunctions)} conjunctions with physics...")
        
        results = []
        for conj in tqdm(potential_conjunctions, desc="Physics Analysis"):
            analysis = self.analyze_conjunction(conj['obj1'], conj['obj2'], conj['screening_epoch'])
            if analysis:
                results.append(analysis)
        
        if not results:
            print("No significant collision risks found")
            return pd.DataFrame()
        
        df = pd.DataFrame(results)
        df = df[df['miss_distance_km'] < 200]  # CHANGED: 50km → 200km
        
        if df.empty:
            print("No valid collision risks")
            return pd.DataFrame()
        
        print(f"\n✓ Phase 1 Complete: {len(df)} conjunctions analyzed with physics")
        print(f"  Max Pc: {df['pc_maximum'].max():.2e}")
        print(f"  Min miss distance: {df['miss_distance_km'].min():.2f} km")
        
        emergency = (df['pc_maximum'] > 1e-4).sum()
        high = ((df['pc_maximum'] > 1e-5) & (df['pc_maximum'] <= 1e-4)).sum()
        medium = ((df['pc_maximum'] > 1e-7) & (df['pc_maximum'] <= 1e-5)).sum()
        low = (df['pc_maximum'] <= 1e-7).sum()
        
        print(f"\nPhysics-Based Risk Distribution:")
        if emergency > 0: print(f"  EMERGENCY: {emergency}")
        if high > 0: print(f"  HIGH: {high}")
        if medium > 0: print(f"  MEDIUM: {medium}")
        print(f"  LOW: {low}")
        
        # PHASE 2: ML TRAINING
        print("\n" + "="*70)
        print("PHASE 2: ML MODEL TRAINING")
        print("="*70)
        
        if not self.ml_predictor.is_trained and len(df) > 0:
            print(f"\nTraining ML model with {len(df)} real conjunctions...")
            self.ml_predictor.train(real_data=df.copy())
        
        # PHASE 3: ADD ML PREDICTIONS TO ALL RESULTS
        if self.ml_predictor.is_trained:
            print("\n" + "="*70)
            print("PHASE 3: ADDING ML PREDICTIONS")
            print("="*70)
            
            print(f"\nGenerating ML predictions for all {len(df)} conjunctions...")
            
            for idx, row in tqdm(df.iterrows(), total=len(df), desc="ML Predictions"):
                ml_predictions = self.ml_predictor.predict_risk(row.to_dict())
                for key, value in ml_predictions.items():
                    df.at[idx, key] = value
            
            print("✓ ML predictions added to all results")
            
            if 'ml_risk_category' in df.columns:
                ml_dist = df['ml_risk_category'].value_counts()
                print(f"\nML-Based Risk Distribution:")
                for cat in ['EMERGENCY', 'HIGH', 'MEDIUM', 'LOW']:
                    count = ml_dist.get(cat, 0)
                    if count > 0:
                        print(f"  {cat}: {count}")
                
                if 'category_agreement' in df.columns:
                    agreement = df['category_agreement'].sum()
                    print(f"\nPhysics-ML Agreement: {agreement}/{len(df)} ({agreement/len(df)*100:.1f}%)")
        
        if 'ml_collision_probability' in df.columns:
            df['combined_risk_score'] = (0.6 * df['pc_maximum'] + 0.4 * df['ml_collision_probability'])
            df = df.sort_values('combined_risk_score', ascending=False)
        else:
            df = df.sort_values('pc_maximum', ascending=False)
        
        self._generate_enhanced_report(df)
        self._save_results(df)
        
        return df    
    
    def _generate_enhanced_report(self, df: pd.DataFrame) -> None:
        """Enhanced report with clear physics vs ML comparison"""
        
        print("\n" + "="*70)
        print("COMPREHENSIVE COLLISION ASSESSMENT REPORT")
        print(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print("="*70)
        
        # PHYSICS SUMMARY
        print("\n" + "="*70)
        print("PHYSICS-BASED ASSESSMENT (NASA/ESA Standards)")
        print("="*70)
        
        emergency = df['pc_maximum'] > 1e-4
        high = (df['pc_maximum'] > 1e-5) & (~emergency)
        medium = (df['pc_maximum'] > 1e-7) & (~emergency) & (~high)
        low = df['pc_maximum'] <= 1e-7
        
        if emergency.sum() > 0:
            print(f"   🔴 EMERGENCY (Pc > 1e-4): {emergency.sum()} events")
        if high.sum() > 0:
            print(f"   🟠 HIGH (Pc > 1e-5): {high.sum()} events")
        if medium.sum() > 0:
            print(f"   🟡 MEDIUM (Pc > 1e-7): {medium.sum()} events")
        print(f"   🟢 LOW (Pc ≤ 1e-7): {low.sum()} events")
        
        print(f"\nPhysics Statistics:")
        print(f"   Max Collision Probability: {df['pc_maximum'].max():.2e}")
        print(f"   Min Miss Distance: {df['miss_distance_km'].min():.3f} km")
        print(f"   Mean Relative Velocity: {df['relative_velocity_ms'].mean():.1f} m/s")
        
        # ML SUMMARY (if available)
        if 'ml_risk_category' in df.columns and df['ml_available'].any():
            print("\n" + "="*70)
            print("ML-BASED ASSESSMENT")
            print("="*70)
            
            ml_categories = df['ml_risk_category'].value_counts()
            for category in ['EMERGENCY', 'HIGH', 'MEDIUM', 'LOW']:
                count = ml_categories.get(category, 0)
                if count > 0:
                    emoji = {'EMERGENCY': '🔴', 'HIGH': '🟠', 'MEDIUM': '🟡', 'LOW': '🟢'}
                    print(f"   {emoji[category]} {category}: {count} events")
            
            print(f"\nML Statistics:")
            ml_probs = df[df['ml_available']]['ml_collision_probability']
            if len(ml_probs) > 0:
                print(f"   Max Collision Probability: {ml_probs.max():.2e}")
                print(f"   Mean Confidence: {df[df['ml_available']]['ml_confidence'].mean():.1f}%")
            
            if 'category_agreement' in df.columns:
                print("\n" + "="*70)
                print("PHYSICS vs ML COMPARISON")
                print("="*70)
                
                agreements = df['category_agreement'].sum()
                total = len(df)
                print(f"   Category Agreement: {agreements}/{total} ({agreements/total*100:.1f}%)")
        
        # TOP RISKS - DETAILED COMPARISON
        print("\n" + "="*70)
        print("TOP 10 COLLISION RISKS - DETAILED COMPARISON")
        print("="*70)
        
        for idx, row in df.head(min(10, len(df))).iterrows():
            print(f"\n{'='*70}")
            print(f"#{idx+1}: {row['object1'][:30]} × {row['object2'][:30]}")
            print(f"{'='*70}")
            print(f"TCA: {row['tca'].strftime('%Y-%m-%d %H:%M:%S UTC')}")
            print(f"Miss Distance: {row['miss_distance_km']:.3f} km")
            print(f"Relative Velocity: {row['relative_velocity_ms']:.0f} m/s")
            print(f"Combined Radius: {row['combined_radius_m']:.1f} m")
            
            pc_phys = row['pc_maximum']
            if pc_phys > 1e-4:
                phys_risk = "🔴 EMERGENCY"
            elif pc_phys > 1e-5:
                phys_risk = "🟠 HIGH"
            elif pc_phys > 1e-7:
                phys_risk = "🟡 MEDIUM"
            else:
                phys_risk = "🟢 LOW"
            
            print(f"\nPHYSICS ASSESSMENT:")
            print(f"   Risk Level: {phys_risk}")
            print(f"   Pc (Monte Carlo): {row['pc_monte_carlo']:.2e}")
            print(f"   Pc (Alfano): {row['pc_alfano']:.2e}")
            print(f"   Pc (Maximum): {pc_phys:.2e}")
            
            if row.get('ml_available', False):
                ml_risk = row['ml_risk_category']
                ml_emoji = {'EMERGENCY': '🔴', 'HIGH': '🟠', 'MEDIUM': '🟡', 'LOW': '🟢'}
                
                print(f"\nML ASSESSMENT:")
                print(f"   Risk Level: {ml_emoji.get(ml_risk, '⚪')} {ml_risk}")
                print(f"   Pc (Predicted): {row['ml_collision_probability']:.2e}")
                print(f"   95% CI: [{row['ml_collision_probability_lower']:.2e}, "
                    f"{row['ml_collision_probability_upper']:.2e}]")
                print(f"   Confidence: {row['ml_confidence']:.1f}%")
                print(f"   Uncertainty: {row['ml_uncertainty']}")
                
                if row.get('category_agreement', True):
                    print(f"\n   ✓ AGREES with physics assessment")
                else:
                    print(f"\n   ⚠ DISAGREES with physics assessment")
                    ratio = row.get('probability_ratio', 0)
                    if ratio > 1:
                        print(f"   ML predicts {ratio:.1f}x HIGHER risk")
                    else:
                        print(f"   ML predicts {1/max(ratio, 0.001):.1f}x LOWER risk")
            else:
                print(f"\nML ASSESSMENT: Not available")
    
    def _save_results(self, df: pd.DataFrame) -> None:
        """Save all results to data/latest directory"""
        
        os.makedirs('data/latest', exist_ok=True)
        
        results_filename = f'data/latest/conjunction_assessment.csv'
        df.to_csv(results_filename, index=False)
        print(f"\n✓ Results saved to: {results_filename}")
        
        if self.ml_predictor.is_trained:
            model_filename = f'data/latest/ml_risk_model.pkl'
            self.ml_predictor.save_model(model_filename)
        
        summary_filename = f'data/latest/summary_report.txt'
        with open(summary_filename, 'w') as f:
            f.write(f"Collision Assessment Summary Report\n")
            f.write(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n")
            f.write(f"Total Conjunctions Analyzed: {len(df)}\n")
            
            if len(df) > 0:
                emergency = (df['pc_maximum'] > 1e-4).sum()
                high = ((df['pc_maximum'] > 1e-5) & (df['pc_maximum'] <= 1e-4)).sum()
                medium = ((df['pc_maximum'] > 1e-7) & (df['pc_maximum'] <= 1e-5)).sum()
                
                f.write(f"EMERGENCY Risks: {emergency}\n")
                f.write(f"HIGH Risks: {high}\n")
                f.write(f"MEDIUM Risks: {medium}\n")
                f.write(f"Max Collision Probability: {df['pc_maximum'].max():.2e}\n")
                f.write(f"Min Miss Distance: {df['miss_distance_km'].min():.3f} km\n")
        
        print(f"✓ Summary report saved to: {summary_filename}")

def fetch_github_tles():
    """Fetch TLE data from GitHub repository"""
    
    url = "https://github.com/harshbhanushali36/TRINETRA_PRODUCTION/blob/main/data/latest/tles.json.gz?raw=true"
    
    try:
        print("Fetching TLE data from GitHub...")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        tle_data = json.loads(gzip.decompress(response.content))
        print(f"Successfully fetched {len(tle_data)} TLE entries from GitHub")
        
        return tle_data
        
    except Exception as e:
        print(f"Error fetching TLE data from GitHub: {e}")
        return None

def load_tle_data_from_json(tle_json):
    """Load TLE data from JSON format"""
    
    tle_data = []
    
    try:
        print(f"Processing TLE JSON with {len(tle_json)} entries...")
        
        for i, item in enumerate(tqdm(tle_json, desc="Processing TLEs")):
            if isinstance(item, dict):
                name = None
                line1 = None
                line2 = None
                
                if 'tle_line1' in item and 'tle_line2' in item:
                    line1 = item['tle_line1']
                    line2 = item['tle_line2']
                elif 'TLE_LINE1' in item and 'TLE_LINE2' in item:
                    line1 = item['TLE_LINE1']
                    line2 = item['TLE_LINE2']
                elif 'TLE_LINE0' in item and 'TLE_LINE1' in item and 'TLE_LINE2' in item:
                    line1 = item['TLE_LINE1']
                    line2 = item['TLE_LINE2']
                else:
                    for key, value in item.items():
                        if isinstance(value, str):
                            if value.startswith('1 ') and len(value) > 60:
                                line1 = value
                            elif value.startswith('2 ') and len(value) > 60:
                                line2 = value
                
                for name_field in ['name', 'OBJECT_NAME', 'object_name', 'satellite_name', 'TLE_LINE0']:
                    if name_field in item:
                        name = item[name_field]
                        break
                
                if not name:
                    name = f"OBJECT_{i}"
                
                if line1 and line2:
                    line1_clean = line1.strip()
                    line2_clean = line2.strip()
                    
                    if (line1_clean.startswith('1 ') and len(line1_clean) >= 69 and
                        line2_clean.startswith('2 ') and len(line2_clean) >= 69):
                        tle_data.append({
                            'name': str(name)[:40],
                            'tle_line1': line1_clean,
                            'tle_line2': line2_clean
                        })
        
        print(f"Processed {len(tle_data)} valid TLE entries")
        return tle_data
        
    except Exception as e:
        print(f"Error processing TLE JSON: {e}")
        return []

def main():
    """Main execution - fully automated for GitHub workflow"""
    
    print("="*70)
    print("AUTOMATED COLLISION PREDICTION SYSTEM")
    print("GitHub Workflow Version - 24h Execution Cycle")
    print("="*70)
    print(f"Start Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    
    time_window_hours = 24
    monte_carlo_samples = 50000
    
    print(f"\nConfiguration:")
    print(f"  ML Risk Scoring: ENABLED")
    print(f"  Time Window: {time_window_hours} hours")
    print(f"  Monte Carlo Samples: {monte_carlo_samples:,}")
    print(f"  Screening Distance: 200 km (INCREASED for more conjunctions)")
    print(f"  Data Source: GitHub TLE repository")
    
    predictor = ScientificallyAccurateCollisionPredictor(
        monte_carlo_samples=monte_carlo_samples
    )
    
    if os.path.exists('data/latest'):
        model_files = [f for f in os.listdir('data/latest') if f.startswith('ml_risk_model') and f.endswith('.pkl')]
        if model_files:
            model_path = os.path.join('data/latest', model_files[0])
            if predictor.ml_predictor.load_model(model_path):
                print(f"✓ Loaded existing ML model: {model_files[0]}")
    
    tle_json = fetch_github_tles()
    if not tle_json:
        print("❌ Failed to fetch TLE data from GitHub")
        return
    
    tle_data = load_tle_data_from_json(tle_json)
    if not tle_data:
        print("❌ No valid TLE data found")
        return
    
    predictor.load_catalog(tle_data)
    
    if len(predictor.objects) == 0:
        print("❌ No valid objects to analyze")
        return
    
    print(f"\nStarting automated analysis...")
    results = predictor.run_analysis(time_window_hours=time_window_hours)
    
    if results.empty:
        print("\n✅ No collision risks identified in this analysis cycle")
    else:
        print(f"\n✅ Analysis completed: {len(results)} collision risks identified")
    
    print(f"\nEnd Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("="*70)

if __name__ == "__main__":
    main()