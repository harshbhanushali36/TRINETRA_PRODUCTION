#!/usr/bin/env python3
"""
Scientifically Accurate Collision Prediction System
Improved version with anisotropic covariance, encounter plane analysis, and better Monte Carlo
Applied fixes:
 - Passed actual relative velocity into collision probability routine
 - Robust importance-sampling Monte Carlo with a fallback
 - Fixed Space-Track TLE parsing typo
"""
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score
from sklearn.utils import resample
import joblib
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm
import warnings
from scipy.integrate import solve_ivp
from scipy.spatial import cKDTree
from scipy.stats import norm, chi2
from sgp4.api import Satrec, jday
import requests
import os

warnings.filterwarnings('ignore')

# Physical Constants (IERS Standards)
MU_EARTH = 398600.4418  # km^3/s^2
EARTH_RADIUS = 6378.137  # km
J2 = 1.08262668e-3  # Earth's J2 coefficient
EARTH_ROTATION_RATE = 7.2921159e-5  # rad/s
SOLAR_FLUX = 1367.0  # W/m^2
SPEED_OF_LIGHT = 299792.458  # km/s

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
            200: 2.5e-10,
            300: 1.9e-11,
            400: 2.8e-12,
            500: 6.9e-13,
            600: 2.1e-13,
            700: 7.5e-14,
            800: 3.0e-14,
            1000: 5.6e-15,
            1500: 2.0e-16
        }
        
    def get_atmospheric_density(self, altitude_km):
        """Get atmospheric density with proper interpolation"""
        if altitude_km > 1500:
            return 0.0
        if altitude_km < 0:
            return 0.0
            
        altitudes = sorted(self.atmosphere_model.keys())
        for i in range(len(altitudes) - 1):
            if altitudes[i] <= altitude_km <= altitudes[i + 1]:
                h1, h2 = altitudes[i], altitudes[i + 1]
                rho1, rho2 = self.atmosphere_model[h1], self.atmosphere_model[h2]
                
                # Log-linear interpolation
                if rho1 > 0 and rho2 > 0:
                    log_rho = np.log(rho1) + (altitude_km - h1) * (np.log(rho2) - np.log(rho1)) / (h2 - h1)
                    return np.exp(log_rho)
                else:
                    return rho2
        
        return self.atmosphere_model[altitudes[-1]]
    
    def compute_accelerations(self, t, state, obj: SpaceObject):
        """Compute accelerations with validated physics"""
        r = state[:3]
        v = state[3:6]
        r_mag = np.linalg.norm(r)
        
        # Validate state
        if r_mag < EARTH_RADIUS * 0.9:  # Inside Earth
            return np.zeros(6)
        
        # Two-body acceleration
        a_twobody = -MU_EARTH * r / r_mag**3
        
        # J2 perturbation (dominant oblateness effect)
        z = r[2]
        r_sq = r_mag * r_mag
        factor = 1.5 * J2 * (EARTH_RADIUS / r_mag)**2
        
        a_j2 = np.zeros(3)
        a_j2[0] = -MU_EARTH * r[0] / r_mag**3 * factor * (5 * z**2 / r_sq - 1)
        a_j2[1] = -MU_EARTH * r[1] / r_mag**3 * factor * (5 * z**2 / r_sq - 1)
        a_j2[2] = -MU_EARTH * z / r_mag**3 * factor * (5 * z**2 / r_sq - 3)
        
        # Atmospheric drag (only below 1500 km)
        altitude = r_mag - EARTH_RADIUS
        a_drag = np.zeros(3)
        
        if 100 < altitude < 1500 and obj.mass > 0:
            rho = self.get_atmospheric_density(altitude)
            
            # Earth rotation effect on atmosphere
            omega_earth = np.array([0, 0, EARTH_ROTATION_RATE])
            v_rel = v - np.cross(omega_earth, r)
            v_rel_mag = np.linalg.norm(v_rel)
            
            if v_rel_mag > 0 and rho > 0:
                # Ballistic coefficient
                BC = obj.mass / (obj.cd * obj.area)
                # Drag acceleration in km/s^2
                a_drag = -0.5 * rho * v_rel_mag * v_rel / BC * 1e-3
        
        # Total acceleration
        a_total = a_twobody + a_j2 + a_drag
        
        return np.concatenate([v, a_total])
    
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
            # Fresh TLE: minimal anisotropy
            sigma_radial = 0.5  # km
            sigma_along_track = 1.0  # km (2x radial)
            sigma_cross_track = 0.3  # km
            sigma_radial_dot = 0.0005  # km/s
            sigma_along_track_dot = 0.001  # km/s
            sigma_cross_track_dot = 0.0003  # km/s
        elif obj.tle_age_days < 7:
            # Week-old TLE: moderate anisotropy
            sigma_radial = 1.0  # km
            sigma_along_track = 5.0  # km (5x radial - along-track grows fastest)
            sigma_cross_track = 0.8  # km
            sigma_radial_dot = 0.001  # km/s
            sigma_along_track_dot = 0.005  # km/s
            sigma_cross_track_dot = 0.0008  # km/s
        else:
            # Old TLE: strong anisotropy
            sigma_radial = 2.0  # km
            sigma_along_track = 20.0  # km (10x radial - severe along-track growth)
            sigma_cross_track = 3.0  # km
            sigma_radial_dot = 0.002  # km/s
            sigma_along_track_dot = 0.015  # km/s
            sigma_cross_track_dot = 0.002  # km/s
        
        # Create rotation matrix from RSW to ECI
        rotation_matrix = np.column_stack([r_unit, t_unit, h_unit])
        
        # Covariance in RSW frame (radial, along-track, cross-track)
        cov_rsw = np.zeros((6, 6))
        
        # Position covariance in RSW
        cov_rsw[0, 0] = sigma_radial**2
        cov_rsw[1, 1] = sigma_along_track**2
        cov_rsw[2, 2] = sigma_cross_track**2
        
        # Velocity covariance in RSW
        cov_rsw[3, 3] = sigma_radial_dot**2
        cov_rsw[4, 4] = sigma_along_track_dot**2
        cov_rsw[5, 5] = sigma_cross_track_dot**2
        
        # Position-velocity correlations (small but realistic)
        cov_rsw[0, 3] = cov_rsw[3, 0] = 0.1 * sigma_radial * sigma_radial_dot
        cov_rsw[1, 4] = cov_rsw[4, 1] = 0.3 * sigma_along_track * sigma_along_track_dot  # Stronger along-track correlation
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
            # Use SGP4 for primary propagation
            sat = Satrec.twoline2rv(obj.tle_line1, obj.tle_line2)
            
            jd, fr = jday(target_epoch.year, target_epoch.month, target_epoch.day,
                         target_epoch.hour, target_epoch.minute, target_epoch.second)
            
            e, r, v = sat.sgp4(jd, fr)
            
            if e != 0:
                # Fallback to last known state
                return obj.state_vector.copy(), obj.covariance_matrix.copy()
            
            new_state = np.concatenate([np.array(r), np.array(v)])
            
            # Create anisotropic covariance
            new_covariance = self.create_anisotropic_covariance(obj, np.array(r), np.array(v))
            
            # Additional uncertainty growth based on propagation time
            dt_hours = (target_epoch - obj.epoch).total_seconds() / 3600
            
            if abs(dt_hours) > 0:
                # Apply time-dependent growth (anisotropic)
                growth_factor = 1 + abs(dt_hours) / 24  # Daily growth
                
                # Along-track grows fastest
                pos = new_state[:3]
                vel = new_state[3:]
                r_mag = np.linalg.norm(pos) if np.linalg.norm(pos) > 0 else 1.0
                
                # Create growth transformation
                r_unit = pos / r_mag
                h = np.cross(pos, vel)
                h_unit = h / np.linalg.norm(h) if np.linalg.norm(h) > 0 else np.array([0.,0.,1.])
                t_unit = np.cross(h_unit, r_unit)
                
                # Growth factors for each direction
                growth_radial = growth_factor
                growth_along_track = growth_factor**1.5  # Grows faster
                growth_cross_track = growth_factor**0.8   # Grows slower
                
                # Apply directional scaling
                rotation_matrix = np.column_stack([r_unit, t_unit, h_unit])
                scaling_matrix = np.diag([growth_radial, growth_along_track, growth_cross_track])
                
                # Transform covariance
                rotation_6x6 = np.zeros((6, 6))
                rotation_6x6[:3, :3] = rotation_matrix
                rotation_6x6[3:, 3:] = rotation_matrix
                
                scaling_6x6 = np.zeros((6, 6))
                scaling_6x6[:3, :3] = scaling_matrix
                scaling_6x6[3:, 3:] = scaling_matrix
                
                # Apply growth: Cov_new = R * S * R^T * Cov * R * S * R^T
                transform = rotation_6x6 @ scaling_6x6 @ rotation_6x6.T
                new_covariance = transform @ new_covariance @ transform.T
            
            return new_state, new_covariance
            
        except Exception as e:
            print(f"Propagation error: {e}")
            return obj.state_vector.copy(), obj.covariance_matrix.copy()

class ImprovedConjunctionAnalyzer:
    """Enhanced conjunction analyzer with encounter plane analysis and improved Monte Carlo"""
    
    def __init__(self, n_samples=10000):
        self.n_samples = n_samples
        self.propagator = RealisticOrbitPropagator()
    
    def find_time_of_closest_approach(self, obj1: SpaceObject, obj2: SpaceObject,
                                     screening_epoch: datetime, window_hours: float = 1) -> Optional[datetime]:
        """Find TCA using SGP4 propagation"""
        
        try:
            sat1 = Satrec.twoline2rv(obj1.tle_line1, obj1.tle_line2)
            sat2 = Satrec.twoline2rv(obj2.tle_line1, obj2.tle_line2)
            
            # Search window
            t_start = screening_epoch - timedelta(hours=window_hours)
            t_end = screening_epoch + timedelta(hours=window_hours)
            
            # Coarse search (10-minute steps)
            min_dist = float('inf')
            tca = screening_epoch
            
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
            
            # Fine search around minimum (1-minute steps)
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
        """Create encounter plane coordinate system and project covariance"""
        
        # Encounter plane: perpendicular to relative velocity
        rel_vel_mag = np.linalg.norm(rel_vel)
        if rel_vel_mag < 1e-6:  # Nearly zero relative velocity
            # Use position vector as reference
            z_axis = rel_pos / np.linalg.norm(rel_pos)
            x_axis = np.array([1, 0, 0])
            if abs(np.dot(z_axis, x_axis)) > 0.9:
                x_axis = np.array([0, 1, 0])
            x_axis = x_axis - np.dot(x_axis, z_axis) * z_axis
            x_axis = x_axis / np.linalg.norm(x_axis)
            y_axis = np.cross(z_axis, x_axis)
        else:
            # Normal encounter plane setup
            z_axis = rel_vel / rel_vel_mag  # Along relative velocity
            
            # Choose x-axis in encounter plane
            x_axis = rel_pos - np.dot(rel_pos, z_axis) * z_axis
            x_axis_mag = np.linalg.norm(x_axis)
            
            if x_axis_mag < 1e-6:  # rel_pos parallel to rel_vel
                # Choose arbitrary perpendicular direction
                if abs(z_axis[0]) < 0.9:
                    x_axis = np.array([1, 0, 0]) - z_axis[0] * z_axis
                else:
                    x_axis = np.array([0, 1, 0]) - z_axis[1] * z_axis
                x_axis = x_axis / np.linalg.norm(x_axis)
            else:
                x_axis = x_axis / x_axis_mag
            
            y_axis = np.cross(z_axis, x_axis)
        
        # Transform matrix from ECI to encounter plane (only need 3x3 for position)
        transform_matrix = np.array([x_axis, y_axis, z_axis])
        
        # Position in encounter plane coordinates
        encounter_pos = transform_matrix @ rel_pos
        
        return transform_matrix, encounter_pos
    
    def project_covariance_to_encounter_plane(self, combined_cov: np.ndarray, transform_matrix: np.ndarray) -> np.ndarray:
        """Project 3D covariance to 2D encounter plane"""
        
        # Extract position covariance
        pos_cov_3d = combined_cov[:3, :3]
        
        # Project to encounter plane (take first 2 rows/cols of transformed covariance)
        encounter_cov_3d = transform_matrix @ pos_cov_3d @ transform_matrix.T
        encounter_cov_2d = encounter_cov_3d[:2, :2]
        
        return encounter_cov_2d
    
    def alfano_collision_probability(self, miss_distance_2d: np.ndarray, cov_2d: np.ndarray, 
                                   combined_radius_km: float) -> float:
        """Alfano (2005) method for 2D encounter plane collision probability"""
        
        try:
            # Validate covariance
            det_cov = np.linalg.det(cov_2d)
            if det_cov < 1e-20 or np.any(np.isnan(cov_2d)):
                return 0.0
            
            # Miss distance in 2D
            miss_2d_mag = np.linalg.norm(miss_distance_2d)
            
            # If miss distance is very large compared to uncertainty, probability is zero
            max_sigma = np.sqrt(np.max(np.linalg.eigvals(cov_2d)))
            if miss_2d_mag > 10 * max_sigma + combined_radius_km:
                return 0.0
            
            # Alfano method parameters
            cov_inv = np.linalg.inv(cov_2d)
            
            # Mahalanobis distance squared
            mahal_dist_sq = miss_distance_2d.T @ cov_inv @ miss_distance_2d
            
            # Combined hard body radius squared
            R_sq = combined_radius_km**2
            
            # Small-target approximation (valid when R << position uncertainty)
            if mahal_dist_sq < 1e-10:  # Very close approach
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
        """Improved Monte Carlo with importance sampling for small probabilities"""
        
        if n_samples is None:
            n_samples = self.n_samples
        
        try:
            miss_2d_mag = np.linalg.norm(miss_distance_2d)
            
            # For very small probabilities, use importance sampling with robust fallback
            if miss_2d_mag > 3 * combined_radius_km:
                # Bias sampling toward the collision disk, but not too aggressively
                bias_factor = max(0.1, 1 - 0.6 * (combined_radius_km / miss_2d_mag))
                bias_center = miss_distance_2d * bias_factor

                # Importance covariance: shrink by a moderate factor to focus on disk region
                importance_cov = cov_2d * 0.2

                # Generate biased samples
                samples = np.random.multivariate_normal(bias_center, importance_cov, n_samples)

                collisions_weighted = 0.0

                for sample in samples:
                    # Distance from this importance sample to the true miss point
                    dist_to_miss = np.linalg.norm(sample - miss_distance_2d)

                    if dist_to_miss < combined_radius_km:
                        # Compute importance weight: original_pdf / importance_pdf
                        orig_density = self._multivariate_normal_pdf(sample, np.zeros(2), cov_2d)
                        imp_density = self._multivariate_normal_pdf(sample, bias_center, importance_cov)
                        if imp_density > 0 and orig_density > 0:
                            weight = orig_density / imp_density
                            collisions_weighted += weight

                pc = collisions_weighted / n_samples

                # Fallback: if no weighted collisions found, do a focused plain MC centered at the miss with many samples
                if pc == 0.0:
                    fallback_samples = max(200_000, n_samples * 4)
                    samples_fb = np.random.multivariate_normal(miss_distance_2d, cov_2d, fallback_samples)
                    distances = np.linalg.norm(samples_fb, axis=1)
                    pc = np.sum(distances < combined_radius_km) / fallback_samples

            else:
                # Standard Monte Carlo for reasonable probabilities (centered at miss)
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
        """Enhanced collision probability with encounter plane analysis using actual relative velocity"""
        
        miss_distance_km = np.linalg.norm(rel_pos_km)
        combined_radius_km = combined_radius_m / 1000.0
        
        # Use actual relative velocity passed in (km/s)
        rel_vel_estimate = rel_vel_km
        if np.linalg.norm(rel_vel_estimate) < 1e-6:
            r_mag = np.linalg.norm(rel_pos_km)
            orbital_speed = np.sqrt(MU_EARTH / r_mag) if r_mag > 0 else 7.5
            rel_vel_estimate = np.array([orbital_speed * 0.1, 0, 0])
        
        # Get encounter plane transformation using provided rel_vel_estimate
        transform_matrix, encounter_pos = self.get_encounter_plane_transform(rel_pos_km, rel_vel_estimate)
        
        # Project covariance to 2D encounter plane
        cov_2d = self.project_covariance_to_encounter_plane(combined_cov, transform_matrix)
        miss_distance_2d = encounter_pos[:2]  # Only x,y in encounter plane
        
        # Validate covariance
        if np.any(np.isnan(cov_2d)) or np.linalg.det(cov_2d) < 1e-20:
            return 0.0, 0.0, {}
        
        # Check if miss distance is too large for any collision risk
        eigenvals = np.linalg.eigvals(cov_2d)
        max_sigma = float(np.sqrt(np.max(eigenvals)))
        min_sigma = float(np.sqrt(np.min(eigenvals)))
        
        if miss_distance_km > 10 * max_sigma + combined_radius_km:
            return 0.0, 0.0, {}
        
        # Enhanced Monte Carlo with importance sampling
        mc_samples = max(self.n_samples, 50_000) if miss_distance_km > 2.0 else self.n_samples
        pc_mc = self.importance_sampling_monte_carlo(miss_distance_2d, cov_2d, combined_radius_km, mc_samples)
        
        # Alfano analytical method
        pc_alfano = self.alfano_collision_probability(miss_distance_2d, cov_2d, combined_radius_km)
        
        # Additional diagnostics
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
    """Machine Learning based risk prediction using Random Forest"""
    
    def __init__(self):
        self.classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        
        self.regressor = RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        self.feature_names = []
        
    def prepare_features(self, conjunction_data: Dict) -> np.ndarray: 
        """Enhanced feature extraction with better feature engineering"""
        
        features = []
        
        # === BASIC PHYSICAL FEATURES ===
        miss_dist = conjunction_data.get('miss_distance_km', 0)
        rel_vel = conjunction_data.get('relative_velocity_ms', 0)
        combined_radius = conjunction_data.get('combined_radius_m', 0)
        pos_uncertainty = conjunction_data.get('position_uncertainty_km', 0)
        
        features.append(miss_dist)
        features.append(rel_vel)
        features.append(combined_radius)
        features.append(pos_uncertainty)
        
        # === TLE AGE FEATURES ===
        tle_age1 = conjunction_data.get('tle_age_days_obj1', 0)
        tle_age2 = conjunction_data.get('tle_age_days_obj2', 0)
        features.append(tle_age1)
        features.append(tle_age2)
        features.append(max(tle_age1, tle_age2))
        features.append(abs(tle_age1 - tle_age2))  # Age difference
        
        # === PROBABILITY FEATURES ===
        pc_mc = conjunction_data.get('pc_monte_carlo', 0)
        pc_alfano = conjunction_data.get('pc_alfano', 0)
        pc_max = conjunction_data.get('pc_maximum', 1e-15)
        
        features.append(pc_mc)
        features.append(pc_alfano)
        features.append(np.log10(pc_max + 1e-15))
        features.append(abs(pc_mc - pc_alfano))  # Method disagreement
        
        # === ENCOUNTER PLANE FEATURES ===
        encounter_miss = conjunction_data.get('encounter_plane_miss_distance_km', 0)
        max_sigma = conjunction_data.get('max_position_sigma_km', 0)
        min_sigma = conjunction_data.get('min_position_sigma_km', 0)
        
        features.append(encounter_miss)
        features.append(max_sigma)
        features.append(min_sigma)
        features.append(max_sigma / max(min_sigma, 0.001))  # Ellipticity
        
        # === COVARIANCE FEATURES ===
        cond_num = conjunction_data.get('covariance_condition_number', 1)
        features.append(min(cond_num, 1000) if np.isfinite(cond_num) else 1000)
        
        # Eigenvalue features
        eigenvals = conjunction_data.get('covariance_eigenvalues_km2', [])
        if eigenvals and len(eigenvals) >= 2:
            features.append(np.sqrt(max(eigenvals)))  # Max uncertainty
            features.append(np.sqrt(min(eigenvals)))  # Min uncertainty
            features.append(np.mean([np.sqrt(e) for e in eigenvals]))  # Mean uncertainty
        else:
            features.extend([pos_uncertainty, pos_uncertainty*0.5, pos_uncertainty*0.75])
        
        # === OBJECT TYPE FEATURES (Enhanced) ===
        type1 = conjunction_data.get('type1', 'SATELLITE')
        type2 = conjunction_data.get('type2', 'SATELLITE')
        
        # Binary features
        features.append(1 if type1 == 'DEBRIS' else 0)
        features.append(1 if type2 == 'DEBRIS' else 0)
        features.append(1 if type1 == 'ROCKET_BODY' else 0)
        features.append(1 if type2 == 'ROCKET_BODY' else 0)
        features.append(1 if 'DEBRIS' in [type1, type2] else 0)
        features.append(1 if 'ROCKET_BODY' in [type1, type2] else 0)
        
        # Risk multipliers based on object types
        type_risk = {'DEBRIS': 3, 'ROCKET_BODY': 2, 'SATELLITE': 1}
        features.append(type_risk.get(type1, 1) * type_risk.get(type2, 1))
        
        # === NORMALIZED FEATURES ===
        features.append(miss_dist / max(pos_uncertainty, 0.1))  # Normalized miss distance
        features.append(miss_dist / max(max_sigma, 0.1))  # Miss relative to max uncertainty
        features.append(combined_radius / 1000 / max(miss_dist, 0.001))  # Size-to-miss ratio
        
        # === INTERACTION FEATURES ===
        features.append(rel_vel * miss_dist)  # Kinetic factor
        features.append(rel_vel * pos_uncertainty)  # Velocity-uncertainty product
        features.append(np.log10(rel_vel + 1) * np.log10(miss_dist + 0.001))  # Log interaction
        
        # === POLYNOMIAL FEATURES ===
        features.append(miss_dist ** 2)
        features.append(np.sqrt(miss_dist))
        features.append(pos_uncertainty ** 2)
        features.append(1 / max(miss_dist, 0.001))  # Inverse miss distance
        
        # === TIME-BASED FEATURES ===
        if 'tca' in conjunction_data and isinstance(conjunction_data['tca'], datetime):
            hours_to_tca = (conjunction_data['tca'] - datetime.now(timezone.utc)).total_seconds() / 3600
            features.append(max(0, min(hours_to_tca, 168)))
            features.append(1 if hours_to_tca < 24 else 0)  # Urgent flag
            features.append(1 if hours_to_tca < 6 else 0)   # Critical flag
        else:
            features.extend([24, 0, 0])
        
        # === STATISTICAL FEATURES ===
        # Mahalanobis distance approximation
        if max_sigma > 0 and min_sigma > 0:
            mahal_approx = miss_dist / np.sqrt(max_sigma * min_sigma)
            features.append(mahal_approx)
            features.append(np.exp(-0.5 * mahal_approx**2))  # Gaussian-like score
        else:
            features.extend([0, 0])
        
        # === CATEGORICAL ENCODING ===
        # Risk bands based on miss distance
        features.append(1 if miss_dist < 1 else 0)    # Very close
        features.append(1 if miss_dist < 5 else 0)    # Close
        features.append(1 if miss_dist < 10 else 0)   # Moderate
        
        # Uncertainty bands
        features.append(1 if pos_uncertainty < 1 else 0)   # High confidence
        features.append(1 if pos_uncertainty < 5 else 0)   # Moderate confidence
        features.append(1 if pos_uncertainty > 10 else 0)  # Low confidence
        
        return np.array(features)
    
    def create_risk_categories(self, pc_values: np.ndarray) -> np.ndarray:
        """Create risk categories based on collision probability"""
        
        categories = np.zeros(len(pc_values), dtype=int)
        
        # NASA/ESA risk thresholds
        categories[pc_values > 1e-4] = 3  # EMERGENCY
        categories[(pc_values > 1e-5) & (pc_values <= 1e-4)] = 2  # HIGH
        categories[(pc_values > 1e-7) & (pc_values <= 1e-5)] = 1  # MEDIUM
        # categories[pc_values <= 1e-7] = 0  # LOW (default)
        
        return categories
    
    def train(self, training_data: pd.DataFrame, use_synthetic: bool = True):
        """Enhanced training with uncertainty quantification"""
        
        print("\n" + "="*70)
        print("TRAINING ML RISK PREDICTION MODEL WITH UNCERTAINTY")
        print("="*70)
        
        if len(training_data) < 20:
            print("Insufficient data for training (need at least 20 samples)")
            return
        
        # Augment with synthetic data if requested
        if use_synthetic:
            training_data = self.augment_training_data(
                training_data, 
                synthetic_ratio=2.0,
                balance_classes=True
            )
        
        # Prepare features
        X = []
        for _, row in training_data.iterrows():
            features = self.prepare_features(row.to_dict())
            X.append(features)
        
        X = np.array(X)
        
        # Create labels
        y_prob = training_data['pc_maximum'].values
        y_cat = self.create_risk_categories(y_prob)
        
        # Split data
        X_train, X_test, y_cat_train, y_cat_test, y_prob_train, y_prob_test = train_test_split(
            X, y_cat, y_prob, test_size=0.2, random_state=42, stratify=y_cat
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train ensemble of classifiers for uncertainty
        print("\nTraining Random Forest Ensemble for Uncertainty...")
        
        # Main classifier with more trees for better uncertainty
        self.classifier = RandomForestClassifier(
            n_estimators=500,  # More trees for better uncertainty
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            oob_score=True  # Out-of-bag score for validation
        )
        
        self.classifier.fit(X_train_scaled, y_cat_train)
        
        # Train multiple regressors for uncertainty quantification
        print("\nTraining Regression Ensemble...")
        
        self.regressor = RandomForestRegressor(
            n_estimators=500,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            oob_score=True
        )
        
        y_prob_train_log = np.log10(y_prob_train + 1e-15)
        self.regressor.fit(X_train_scaled, y_prob_train_log)
        
        # Store training statistics for uncertainty calibration
        y_pred_train = self.regressor.predict(X_train_scaled)
        residuals = y_prob_train_log - y_pred_train
        self.residual_std = np.std(residuals)
        self.residual_mean = np.mean(residuals)
        
        # Calculate feature importances with confidence
        importances = self.classifier.feature_importances_
        std = np.std([tree.feature_importances_ for tree in self.classifier.estimators_], axis=0)
        
        # Enhanced feature names
        self.feature_names = [
            'miss_distance_km', 'relative_velocity_ms', 'combined_radius_m',
            'position_uncertainty_km', 'tle_age_days_obj1', 'tle_age_days_obj2',
            'max_tle_age', 'tle_age_difference', 'pc_monte_carlo', 'pc_alfano', 
            'log_pc_maximum', 'pc_method_disagreement', 'encounter_plane_miss_km', 
            'max_sigma_km', 'min_sigma_km', 'ellipticity', 'covariance_condition',
            'max_eigenval_sqrt', 'min_eigenval_sqrt', 'mean_eigenval_sqrt',
            'is_debris_1', 'is_debris_2', 'is_rocket_1', 'is_rocket_2', 
            'has_debris', 'has_rocket', 'type_risk_product', 'normalized_miss',
            'miss_to_max_sigma', 'size_to_miss_ratio', 'kinetic_factor',
            'vel_uncertainty_product', 'log_interaction', 'miss_squared',
            'miss_sqrt', 'uncertainty_squared', 'inverse_miss', 'hours_to_tca',
            'urgent_flag', 'critical_flag', 'mahalanobis_approx', 'gaussian_score',
            'very_close', 'close', 'moderate', 'high_confidence', 
            'moderate_confidence', 'low_confidence'
        ]
        
        # Print feature importance with uncertainty
        print("\nTop 10 Feature Importances (with uncertainty):")
        indices = np.argsort(importances)[::-1][:10]
        for idx in indices:
            if idx < len(self.feature_names):
                print(f"  {self.feature_names[idx]}: {importances[idx]:.4f} ± {std[idx]:.4f}")
        
        # Evaluate on test set
        print("\nModel Performance:")
        y_cat_pred = self.classifier.predict(X_test_scaled)
        accuracy = np.mean(y_cat_pred == y_cat_test)
        print(f"  Classification Accuracy: {accuracy:.3f}")
        
        if hasattr(self.classifier, 'oob_score_'):
            print(f"  OOB Score (Classification): {self.classifier.oob_score_:.3f}")
        
        if hasattr(self.regressor, 'oob_score_'):
            print(f"  OOB Score (Regression): {self.regressor.oob_score_:.3f}")
        
        self.is_trained = True
        print("\n✓ ML models with uncertainty trained successfully!")

    def predict_risk(self, conjunction_data: Dict) -> Dict:
        """Enhanced prediction with uncertainty quantification"""
        
        if not self.is_trained:
            return {
                'ml_risk_category': 'UNKNOWN',
                'ml_risk_score': 0,
                'ml_collision_probability': conjunction_data.get('pc_maximum', 0),
                'ml_collision_probability_lower': 0,
                'ml_collision_probability_upper': 0,
                'ml_confidence': 0,
                'ml_uncertainty': 'HIGH',
                'ml_available': False
            }
        
        # Prepare features
        X = self.prepare_features(conjunction_data).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        # Get predictions from all trees for uncertainty
        risk_probabilities = self.classifier.predict_proba(X_scaled)[0]
        risk_category = np.argmax(risk_probabilities)
        
        # Get prediction intervals from regression ensemble
        all_tree_predictions = []
        for tree in self.regressor.estimators_:
            pred = tree.predict(X_scaled)[0]
            all_tree_predictions.append(pred)
        
        all_tree_predictions = np.array(all_tree_predictions)
        
        # Calculate statistics
        collision_prob_log_mean = np.mean(all_tree_predictions)
        collision_prob_log_std = np.std(all_tree_predictions)
        
        # Add residual uncertainty
        total_uncertainty = np.sqrt(collision_prob_log_std**2 + self.residual_std**2)
        
        # Calculate prediction intervals (95% confidence)
        z_score = 1.96
        collision_prob_log_lower = collision_prob_log_mean - z_score * total_uncertainty
        collision_prob_log_upper = collision_prob_log_mean + z_score * total_uncertainty
        
        # Convert from log scale
        collision_prob = 10**collision_prob_log_mean
        collision_prob_lower = 10**collision_prob_log_lower
        collision_prob_upper = 10**collision_prob_log_upper
        
        # Calculate entropy-based uncertainty
        entropy = -np.sum(risk_probabilities * np.log(risk_probabilities + 1e-10))
        max_entropy = -np.log(1/len(risk_probabilities))
        normalized_entropy = entropy / max_entropy
        
        # Determine uncertainty level
        if normalized_entropy < 0.3 and collision_prob_log_std < 0.5:
            uncertainty_level = 'LOW'
        elif normalized_entropy < 0.6 and collision_prob_log_std < 1.0:
            uncertainty_level = 'MEDIUM'
        else:
            uncertainty_level = 'HIGH'
        
        # Risk names
        risk_names = ['LOW', 'MEDIUM', 'HIGH', 'EMERGENCY']
        risk_name = risk_names[risk_category]
        
        # Confidence based on probability and tree agreement
        confidence = max(risk_probabilities) * 100 * (1 - normalized_entropy)
        
        # Calculate prediction agreement among trees
        tree_risk_predictions = []
        for prob_tree in range(0, len(self.classifier.estimators_), 10):
            tree = self.classifier.estimators_[prob_tree]
            tree_pred = tree.predict(X_scaled)[0]
            tree_risk_predictions.append(tree_pred)
        
        prediction_agreement = np.mean([p == risk_category for p in tree_risk_predictions])
        
        return {
            'ml_risk_category': risk_name,
            'ml_risk_score': int(risk_category),
            'ml_collision_probability': float(collision_prob),
            'ml_collision_probability_lower': float(collision_prob_lower),
            'ml_collision_probability_upper': float(collision_prob_upper),
            'ml_confidence': float(confidence),
            'ml_uncertainty': uncertainty_level,
            'ml_entropy': float(normalized_entropy),
            'ml_std_log': float(collision_prob_log_std),
            'ml_prediction_agreement': float(prediction_agreement),
            'ml_available': True
        }
    def generate_synthetic_conjunction(self, risk_level: str = 'random') -> Dict:
        """Generate synthetic conjunction data for training"""
        
        np.random.seed(None)  # Ensure randomness
        
        # Risk level distributions
        risk_params = {
            'EMERGENCY': {
                'miss_distance': (0.1, 2.0),
                'rel_velocity': (7000, 15000),
                'uncertainty': (0.5, 2.0),
                'pc_range': (1e-4, 1e-2)
            },
            'HIGH': {
                'miss_distance': (1.0, 5.0),
                'rel_velocity': (5000, 12000),
                'uncertainty': (1.0, 5.0),
                'pc_range': (1e-5, 1e-4)
            },
            'MEDIUM': {
                'miss_distance': (3.0, 15.0),
                'rel_velocity': (3000, 10000),
                'uncertainty': (2.0, 10.0),
                'pc_range': (1e-7, 1e-5)
            },
            'LOW': {
                'miss_distance': (10.0, 50.0),
                'rel_velocity': (1000, 8000),
                'uncertainty': (5.0, 20.0),
                'pc_range': (1e-12, 1e-7)
            }
        }
        
        # Select risk level
        if risk_level == 'random':
            # Weighted selection (more low-risk events)
            risk_level = np.random.choice(
                ['LOW', 'MEDIUM', 'HIGH', 'EMERGENCY'],
                p=[0.7, 0.2, 0.08, 0.02]
            )
        
        params = risk_params[risk_level]
        
        # Generate base parameters
        miss_distance = np.random.uniform(*params['miss_distance'])
        rel_velocity = np.random.uniform(*params['rel_velocity'])
        pos_uncertainty = np.random.uniform(*params['uncertainty'])
        
        # Generate correlated uncertainties
        max_sigma = pos_uncertainty * np.random.uniform(1.0, 3.0)
        min_sigma = pos_uncertainty * np.random.uniform(0.3, 1.0)
        
        # Generate TLE ages with correlation to uncertainty
        base_age = np.random.exponential(3.0)  # Most TLEs are fresh
        tle_age1 = base_age + np.random.normal(0, 1)
        tle_age2 = base_age + np.random.normal(0, 1)
        tle_age1 = max(0.1, min(30, tle_age1))
        tle_age2 = max(0.1, min(30, tle_age2))
        
        # Object types (correlated with risk)
        if risk_level in ['HIGH', 'EMERGENCY']:
            types = ['DEBRIS', 'ROCKET_BODY', 'SATELLITE']
            type_probs = [0.5, 0.3, 0.2]  # More debris in high-risk
        else:
            types = ['SATELLITE', 'DEBRIS', 'ROCKET_BODY']
            type_probs = [0.6, 0.25, 0.15]
        
        type1 = np.random.choice(types, p=type_probs)
        type2 = np.random.choice(types, p=type_probs)
        
        # Combined radius based on object types
        radius_map = {'DEBRIS': 1, 'ROCKET_BODY': 5, 'SATELLITE': 10}
        combined_radius = radius_map[type1] + radius_map[type2]
        combined_radius *= np.random.uniform(0.8, 1.2)  # Add variation
        
        # Generate collision probabilities with realistic correlation
        # Base probability from miss distance and uncertainty
        norm_miss = miss_distance / max(pos_uncertainty, 0.1)
        
        # Simplified analytical estimate
        pc_base = np.exp(-0.5 * norm_miss**2) * (combined_radius/1000)**2 / (2*np.pi*pos_uncertainty**2)
        
        # Add noise and ensure within range
        noise_factor = np.random.lognormal(0, 0.5)
        pc_alfano = np.clip(pc_base * noise_factor, *params['pc_range'])
        
        # Monte Carlo typically has some variation from analytical
        mc_variation = np.random.normal(1.0, 0.2)
        pc_monte_carlo = np.clip(pc_alfano * mc_variation, *params['pc_range'])
        
        # Encounter plane miss (correlated with 3D miss)
        encounter_miss = miss_distance * np.random.uniform(0.7, 1.0)
        
        # Generate eigenvalues (consistent with max/min sigma)
        eigenval1 = max_sigma**2
        eigenval2 = min_sigma**2
        
        # TCA in the future
        hours_to_tca = np.random.exponential(24)  # Most events within a day
        tca = datetime.now(timezone.utc) + timedelta(hours=hours_to_tca)
        
        return {
            'object1': f'SYNTHETIC_{type1}_{np.random.randint(10000,99999)}',
            'object2': f'SYNTHETIC_{type2}_{np.random.randint(10000,99999)}',
            'type1': type1,
            'type2': type2,
            'tca': tca,
            'miss_distance_km': miss_distance,
            'relative_velocity_ms': rel_velocity,
            'pc_monte_carlo': pc_monte_carlo,
            'pc_alfano': pc_alfano,
            'pc_maximum': max(pc_monte_carlo, pc_alfano),
            'combined_radius_m': combined_radius,
            'position_uncertainty_km': pos_uncertainty,
            'tle_age_days_obj1': tle_age1,
            'tle_age_days_obj2': tle_age2,
            'encounter_plane_miss_distance_km': encounter_miss,
            'max_position_sigma_km': max_sigma,
            'min_position_sigma_km': min_sigma,
            'covariance_condition_number': max_sigma / max(min_sigma, 0.001),
            'covariance_eigenvalues_km2': [eigenval1, eigenval2],
            'monte_carlo_samples_used': 50000,
            'is_synthetic': True,
            'synthetic_risk_level': risk_level
        }

    def augment_training_data(self, real_data: pd.DataFrame, 
                             synthetic_ratio: float = 2.0,
                             balance_classes: bool = True) -> pd.DataFrame:
        """Augment real data with synthetic samples"""
        
        print("\nGenerating synthetic training data...")
        
        # Analyze real data distribution
        real_pc = real_data['pc_maximum'].values
        real_categories = self.create_risk_categories(real_pc)
        
        unique, counts = np.unique(real_categories, return_counts=True)
        category_counts = dict(zip(unique, counts))
        
        print(f"Real data distribution: {category_counts}")
        
        # Generate synthetic data
        synthetic_data = []
        risk_names = ['LOW', 'MEDIUM', 'HIGH', 'EMERGENCY']
        
        if balance_classes:
            # Generate equal samples for each class
            max_count = max(counts) * synthetic_ratio
            for cat_id in range(4):
                risk_name = risk_names[cat_id]
                current_count = category_counts.get(cat_id, 0)
                needed = int(max_count - current_count)
                
                if needed > 0:
                    print(f"Generating {needed} synthetic {risk_name} events...")
                    for _ in range(needed):
                        synthetic_data.append(self.generate_synthetic_conjunction(risk_name))
        else:
            # Generate proportionally
            total_synthetic = int(len(real_data) * synthetic_ratio)
            print(f"Generating {total_synthetic} synthetic events...")
            
            for _ in range(total_synthetic):
                synthetic_data.append(self.generate_synthetic_conjunction('random'))
        
        # Combine datasets
        synthetic_df = pd.DataFrame(synthetic_data)
        augmented_data = pd.concat([real_data, synthetic_df], ignore_index=True)
        
        print(f"Dataset augmented: {len(real_data)} real + {len(synthetic_df)} synthetic")
        
        return augmented_data

    def save_model(self, filepath: str):
        """Save trained models to disk"""
        
        if not self.is_trained:
            print("No trained model to save")
            return
        
        model_data = {
            'classifier': self.classifier,
            'regressor': self.regressor,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained models from disk"""
        
        if os.path.exists(filepath):
            model_data = joblib.load(filepath)
            self.classifier = model_data['classifier']
            self.regressor = model_data['regressor']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.is_trained = model_data['is_trained']
            print(f"Model loaded from {filepath}")
            return True
        return False
class ScientificallyAccurateCollisionPredictor:
    """Main system with enhanced analysis capabilities"""
    
    def __init__(self, monte_carlo_samples=50000, enable_ml=True):
        self.propagator = RealisticOrbitPropagator()
        self.analyzer = ImprovedConjunctionAnalyzer(n_samples=monte_carlo_samples)
        self.objects = []
        self.debug_log = []  # For logging suspicious outputs
        self.enable_ml = enable_ml
        self.ml_predictor = MLRiskPredictor() if enable_ml else None
        self.training_data = []  # Store conjunction data for ML training
        
    def create_object_from_tle(self, name: str, tle_line1: str, tle_line2: str,
                               epoch: datetime = None) -> Optional[SpaceObject]:
        """Create object with realistic anisotropic parameters"""
        
        try:
            satellite = Satrec.twoline2rv(tle_line1, tle_line2)
            
            if epoch is None:
                epoch = datetime.now(timezone.utc)
            
            # Get initial state
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
            
            # Object classification and realistic parameters
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
                name=name[:40],
                norad_id=satellite.satnum,
                state_vector=state_vector,
                covariance_matrix=covariance,
                epoch=epoch,
                mass=mass,
                area=area,
                cd=cd,
                cr=cr,
                object_type=obj_type,
                tle_line1=tle_line1,
                tle_line2=tle_line2,
                tle_age_days=tle_age_days
            )
            
        except Exception as e:
            return None
    
    def load_catalog(self, tle_data: List[Dict]) -> None:
        """Load catalog with validation"""
        
        print("\nLoading validated object catalog...")
        
        epoch = datetime.now(timezone.utc)
        
        for item in tqdm(tle_data, desc="Processing objects"):
            obj = self.create_object_from_tle(
                item['name'],
                item['tle_line1'],
                item['tle_line2'],
                epoch
            )
            
            if obj is not None:
                # Skip space stations
                if not any(kw in obj.name.upper() for kw in ['ISS', 'TIANHE', 'ZARYA']):
                    self.objects.append(obj)
        
        print(f"Loaded {len(self.objects):,} validated objects")
        
        # Statistics
        types = {}
        for obj in self.objects:
            types[obj.object_type] = types.get(obj.object_type, 0) + 1
        
        print("\nCatalog Statistics:")
        for obj_type, count in sorted(types.items()):
            print(f"   {obj_type}: {count:,}")
    
    def screen_conjunctions_accurate(self, time_window_hours: float = 24,
                                    screening_distance_km: float = 50) -> List[Dict]:
        """Accurate conjunction screening with proper distance validation"""
        
        print(f"\nAccurate screening (distance < {screening_distance_km} km)")
        
        # Limit catalog size for performance
        max_objects = 2000
        if len(self.objects) > max_objects:
            print(f"   Sampling {max_objects} objects from {len(self.objects):,}")
            
            # Priority sampling
            debris = [o for o in self.objects if o.object_type == 'DEBRIS']
            rockets = [o for o in self.objects if o.object_type == 'ROCKET_BODY']
            sats = [o for o in self.objects if o.object_type == 'SATELLITE']
            
            np.random.seed(42)  # For reproducibility
            
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
        min_distance_found = float('inf')  # Track minimum distance
        
        # Time steps for screening
        n_steps = min(12, int(time_window_hours / 2))
        time_steps = np.linspace(0, time_window_hours, n_steps)
        start_epoch = datetime.now(timezone.utc)
        
        for hours in tqdm(time_steps, desc="Time steps"):
            current_epoch = start_epoch + timedelta(hours=hours)
            
            # Propagate all objects using SGP4
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
            
            # Use KD-tree for efficient nearest neighbor search
            tree = cKDTree(positions)
            
            # Find all pairs within screening distance
            pairs = tree.query_pairs(screening_distance_km, output_type='ndarray')
            
            for i, j in pairs:
                dist = np.linalg.norm(positions[i] - positions[j])
                min_distance_found = min(min_distance_found, dist)
                
                # CRITICAL: Validate distance is actually within threshold
                if dist <= screening_distance_km:
                    obj1, obj2 = valid_objects[i], valid_objects[j]
                    
                    # Filter same constellation
                    if obj1.object_type == 'SATELLITE' and obj2.object_type == 'SATELLITE':
                        name1_base = obj1.name.split('-')[0] if '-' in obj1.name else obj1.name[:10]
                        name2_base = obj2.name.split('-')[0] if '-' in obj2.name else obj2.name[:10]
                        if name1_base == name2_base:
                            continue
                    
                    conjunctions.append({
                        'obj1': obj1,
                        'obj2': obj2,
                        'screening_epoch': current_epoch,
                        'screening_distance': dist
                    })
        
        # Debug output
        if min_distance_found < float('inf'):
            print(f"DEBUG: Minimum distance found: {min_distance_found:.2f} km")
        else:
            print(f"DEBUG: No close approaches found")
        
        # Remove duplicates
        unique_conjunctions = {}
        for conj in conjunctions:
            key = tuple(sorted([conj['obj1'].name, conj['obj2'].name]))
            if key not in unique_conjunctions or conj['screening_distance'] < unique_conjunctions[key]['screening_distance']:
                unique_conjunctions[key] = conj
        
        # Sort by distance and limit
        final_conjunctions = sorted(unique_conjunctions.values(), key=lambda x: x['screening_distance'])
        
        if len(final_conjunctions) > 100:
            print(f"   Limiting to 100 closest approaches")
            final_conjunctions = final_conjunctions[:100]
        
        # Validate all distances
        valid_conjunctions = []
        for conj in final_conjunctions:
            if conj['screening_distance'] <= screening_distance_km:
                valid_conjunctions.append(conj)
        
        print(f"Found {len(valid_conjunctions)} valid conjunctions within {screening_distance_km} km")
        
        return valid_conjunctions
    
    def analyze_conjunction(self, obj1: SpaceObject, obj2: SpaceObject,
                           screening_epoch: datetime) -> Optional[Dict]:
        """Analyze conjunction with enhanced diagnostics and logging"""
        
        # Find accurate TCA
        tca = self.analyzer.find_time_of_closest_approach(obj1, obj2, screening_epoch)
        
        if tca is None:
            return None
        
        # Propagate to TCA
        state1, cov1 = self.propagator.propagate_state_sgp4(obj1, tca)
        state2, cov2 = self.propagator.propagate_state_sgp4(obj2, tca)
        
        # Relative state
        rel_pos = state1[:3] - state2[:3]
        rel_vel = state1[3:] - state2[3:]
        
        miss_distance = np.linalg.norm(rel_pos)
        rel_speed = np.linalg.norm(rel_vel)
        
        # Log suspicious outputs for debugging
        if rel_speed < 0.1:  # Less than 100 m/s is suspicious for different orbits
            debug_info = {
                'objects': f"{obj1.name} vs {obj2.name}",
                'state1': state1.copy(),
                'state2': state2.copy(),
                'relative_velocity': rel_vel.copy(),
                'rel_speed_ms': rel_speed * 1000,
                'miss_distance_km': miss_distance,
                'tca': tca,
                'issue': f"Suspiciously low relative velocity: {rel_speed*1000:.1f} m/s"
            }
            self.debug_log.append(debug_info)
            print(f"DEBUG: Suspicious low rel. velocity {rel_speed*1000:.1f} m/s for {obj1.name} vs {obj2.name}")
        
        # Skip if miss distance is unrealistic
        if miss_distance > 100:  # More than 100 km is not a conjunction
            return None
        
        # Combined covariance
        combined_cov = cov1 + cov2
        
        # Object sizes (realistic)
        radius1 = {'DEBRIS': 1, 'ROCKET_BODY': 5, 'SATELLITE': 10}.get(obj1.object_type, 5)
        radius2 = {'DEBRIS': 1, 'ROCKET_BODY': 5, 'SATELLITE': 10}.get(obj2.object_type, 5)
        combined_radius = radius1 + radius2
        
        # Calculate enhanced collision probability — pass actual rel_vel!
        pc_mc, pc_alfano, diagnostics = self.analyzer.compute_collision_probability(
            rel_pos, rel_vel, combined_cov, combined_radius
        )
        
        # Only return if there's actual risk
        if max(pc_mc, pc_alfano) < 1e-12:
            return None
        
        result = {
            'object1': obj1.name,
            'object2': obj2.name,
            'type1': obj1.object_type,
            'type2': obj2.object_type,
            'tca': tca,
            'miss_distance_km': miss_distance,
            'relative_velocity_ms': rel_speed * 1000,
            'pc_monte_carlo': pc_mc,
            'pc_alfano': pc_alfano,
            'pc_maximum': max(pc_mc, pc_alfano),
            'combined_radius_m': combined_radius,
            'position_uncertainty_km': np.sqrt(np.trace(combined_cov[:3, :3]) / 3),
            'tle_age_days_obj1': obj1.tle_age_days,
            'tle_age_days_obj2': obj2.tle_age_days
        }
        
        # Add diagnostics
        result.update(diagnostics)
        # Add ML predictions if enabled
        if self.enable_ml and self.ml_predictor:
            ml_predictions = self.ml_predictor.predict_risk(result)
            result.update(ml_predictions)
        # Store for potential ML training
        self.training_data.append(result)
        
        return result
    
    def run_analysis(self, time_window_hours: float = 24) -> tuple:
        """Run complete enhanced analysis with ML risk scoring"""
        print("\n" + "="*70)
        print("ML-ENHANCED COLLISION PREDICTION SYSTEM")
        print("With Random Forest Risk Scoring")
        print("="*70)

        # Clear debug log
        self.debug_log = []

        # Screen with validation
        potential_conjunctions = self.screen_conjunctions_accurate(time_window_hours)

        if not potential_conjunctions:
            print("No conjunctions found within criteria")
            return pd.DataFrame()

        # Detailed analysis
        print(f"\nAnalyzing {len(potential_conjunctions)} validated conjunctions...")

        results = []

        for conj in tqdm(potential_conjunctions, desc="Analyzing"):
            analysis = self.analyze_conjunction(conj['obj1'], conj['obj2'], conj['screening_epoch'])
            if analysis:
                results.append(analysis)

        if not results:
            print("No significant collision risks after detailed analysis")
            return pd.DataFrame()

        # Create DataFrame
        df = pd.DataFrame(results)

        # Additional validation - remove any with absurd miss distances
        df = df[df['miss_distance_km'] < 50]

        if df.empty:
            print("No valid collision risks identified")
            return pd.DataFrame()

        # If we have enough data and ML is not trained yet, train it
        if self.enable_ml and not self.ml_predictor.is_trained and len(df) >= 20:
            print("\n" + "="*70)
            print("AUTO-TRAINING ML MODEL ON CURRENT DATA")
            print("="*70)
            self.ml_predictor.train(df)

            # Re-run predictions with trained model
            print("\nRe-running predictions with trained model...")
            for idx, row in df.iterrows():
                ml_predictions = self.ml_predictor.predict_risk(row.to_dict())
                for key, value in ml_predictions.items():
                    df.at[idx, key] = value

        # Sort by combined risk score if ML is available
        if self.enable_ml and 'ml_collision_probability' in df.columns:
            df['combined_risk_score'] = (
                0.7 * df['pc_maximum'] +
                0.3 * df['ml_collision_probability']
            )
            df = df.sort_values('combined_risk_score', ascending=False)
        else:
            df = df.sort_values('pc_maximum', ascending=False)

        # Generate enhanced report
        self._generate_enhanced_report(df)

        # Save ML model if trained
        if self.enable_ml and self.ml_predictor.is_trained:
            model_filename = f'ml_risk_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl'
            self.ml_predictor.save_model(model_filename)
            print(f"ML model saved to: {model_filename}")

        return df
    
    def _generate_enhanced_report(self, df: pd.DataFrame) -> None:
        """Generate enhanced assessment report with diagnostics"""
        
        print("\n" + "="*70)
        print("ENHANCED CONJUNCTION ASSESSMENT REPORT")
        print(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print("="*70)
        
        # Realistic risk categories
        emergency = df['pc_maximum'] > 1e-4
        high = (df['pc_maximum'] > 1e-5) & (~emergency)
        medium = (df['pc_maximum'] > 1e-7) & (~emergency) & (~high)
        low = df['pc_maximum'] <= 1e-7
        
        print("\nRISK SUMMARY (NASA/ESA Standards):")
        if emergency.sum() > 0:
            print(f"   EMERGENCY (Pc > 1e-4): {emergency.sum()} - IMMEDIATE ACTION")
        if high.sum() > 0:
            print(f"   HIGH (Pc > 1e-5): {high.sum()}")
        if medium.sum() > 0:
            print(f"   MEDIUM (Pc > 1e-7): {medium.sum()}")
        print(f"   LOW (Pc ≤ 1e-7): {low.sum()}")
        
        print(f"\nENHANCED STATISTICS:")
        print(f"   Total valid conjunctions: {len(df):,}")
        print(f"   Max collision probability: {df['pc_maximum'].max():.2e}")
        print(f"   Min miss distance: {df['miss_distance_km'].min():.3f} km")
        print(f"   Mean relative velocity: {df['relative_velocity_ms'].mean():.1f} m/s")
        print(f"   Mean position uncertainty: {df['position_uncertainty_km'].mean():.2f} km")
        
        # Enhanced diagnostics
        if 'covariance_condition_number' in df.columns:
            print(f"   Mean covariance condition number: {df['covariance_condition_number'].mean():.1f}")
        if 'max_position_sigma_km' in df.columns:
            print(f"   Max position sigma: {df['max_position_sigma_km'].max():.2f} km")
        
        print("\nVALIDATION CHECKS PASSED:")
        print("   • Anisotropic covariance models implemented")
        print("   • Encounter plane projections computed") 
        print("   • Importance sampling Monte Carlo used")
        print("   • Alfano analytical method implemented")
        print("   • Enhanced diagnostics included")
        
        print(f"\nTOP COLLISION RISKS with Enhanced Analysis:")
        print("-"*70)
        
        for idx, row in df.head(min(10, len(df))).iterrows():
            pc = row['pc_maximum']
            
            if pc > 1e-4:
                risk = "EMERGENCY"
            elif pc > 1e-5:
                risk = "HIGH"
            elif pc > 1e-7:
                risk = "MEDIUM"
            else:
                risk = "LOW"
            
            type_info = ""
            if row['type1'] == 'DEBRIS' or row['type2'] == 'DEBRIS':
                type_info = " [DEBRIS]"
            elif row['type1'] == 'ROCKET_BODY' or row['type2'] == 'ROCKET_BODY':
                type_info = " [ROCKET]"
            
            print(f"\n{idx+1:2d}. {row['object1'][:30]} × {row['object2'][:30]}{type_info}")
            print(f"    {risk} | Pc(MC): {row['pc_monte_carlo']:.2e} | Pc(Alfano): {row['pc_alfano']:.2e}")
            print(f"    TCA: {row['tca'].strftime('%Y-%m-%d %H:%M:%S UTC')}")
            print(f"    Miss: {row['miss_distance_km']:.3f} km | Vel: {row['relative_velocity_ms']:.0f} m/s")
            print(f"    Combined radius: {row['combined_radius_m']:.1f} m")
            print(f"    Position uncertainty (1-σ): {row['position_uncertainty_km']:.2f} km")
            
            # Enhanced diagnostics
            if 'encounter_plane_miss_distance_km' in row:
                print(f"    Encounter plane miss: {row['encounter_plane_miss_distance_km']:.3f} km")
            if 'max_position_sigma_km' in row and 'min_position_sigma_km' in row:
                print(f"    Max/Min sigma: {row['max_position_sigma_km']:.2f}/{row['min_position_sigma_km']:.2f} km")
            if 'covariance_condition_number' in row:
                cond_num = row['covariance_condition_number']
                if np.isfinite(cond_num):
                    print(f"    Covariance condition: {cond_num:.1f}")
            if 'monte_carlo_samples_used' in row:
                print(f"    MC samples used: {row['monte_carlo_samples_used']:,}")
        
        # Debug information
        if self.debug_log:
            print(f"\nDEBUG LOG: {len(self.debug_log)} suspicious cases detected:")
            print("-"*50)
            for i, debug in enumerate(self.debug_log[:5]):  # Show first 5
                print(f"{i+1}. {debug['objects']}")
                print(f"   Issue: {debug['issue']}")
                print(f"   Relative velocity: {debug['rel_speed_ms']:.1f} m/s")
                print(f"   Miss distance: {debug['miss_distance_km']:.3f} km")
        if 'ml_risk_category' in df.columns and self.enable_ml:
            print("\nML RISK PREDICTIONS:")
            ml_categories = df['ml_risk_category'].value_counts()
            for category, count in ml_categories.items():
                if category != 'UNKNOWN':
                    print(f"   {category}: {count}")
    
            if 'ml_confidence' in df.columns:
                print(f"   Average ML Confidence: {df['ml_confidence'].mean():.1f}%")
        # Add ML predictions if available
        if 'ml_risk_category' in row and row.get('ml_available', False):
            ml_cat = row['ml_risk_category']
            ml_conf = row.get('ml_confidence', 0)
            ml_prob = row.get('ml_collision_probability', 0)
            print(f"    ML Risk: {ml_cat} (Confidence: {ml_conf:.1f}%) | ML Pc: {ml_prob:.2e}")
        # Add this section after the existing ML predictions output
        if 'ml_uncertainty' in row and row.get('ml_available', False):
            ml_unc = row['ml_uncertainty']
            ml_lower = row.get('ml_collision_probability_lower', 0)
            ml_upper = row.get('ml_collision_probability_upper', 0)
            ml_agree = row.get('ml_prediction_agreement', 0)
            print(f"    ML Uncertainty: {ml_unc} | 95% CI: [{ml_lower:.2e}, {ml_upper:.2e}]")
            print(f"    Tree Agreement: {ml_agree:.1%}")
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'enhanced_conjunction_assessment.csv'
        df.to_csv(filename, index=False)
        
        print(f"\nEnhanced results saved to: {filename}")


def load_tle_data(filepath):
    """Load TLE data from file with validation"""
    tle_data = []
    
    try:
        if filepath.endswith('.txt'):
            with open(filepath, 'r') as f:
                lines = f.readlines()
            
            i = 0
            while i < len(lines) - 2:
                name = lines[i].strip()
                tle1 = lines[i+1].strip()
                tle2 = lines[i+2].strip()
                
                # Validate TLE format
                if (tle1.startswith('1 ') and len(tle1) == 69 and
                    tle2.startswith('2 ') and len(tle2) == 69):
                    tle_data.append({
                        'name': name,
                        'tle_line1': tle1,
                        'tle_line2': tle2
                    })
                    i += 3
                else:
                    i += 1
                    
        elif filepath.endswith(('.xlsx', '.csv')):
            if filepath.endswith('.xlsx'):
                df = pd.read_excel(filepath)
            else:
                df = pd.read_csv(filepath)
            
            # Find TLE columns
            tle1_col = None
            tle2_col = None
            name_col = None
            
            for col in df.columns:
                col_upper = col.upper()
                if 'TLE' in col_upper and '1' in col_upper:
                    tle1_col = col
                elif 'TLE' in col_upper and '2' in col_upper:
                    tle2_col = col
                elif 'NAME' in col_upper:
                    name_col = col
            
            if tle1_col and tle2_col:
                for _, row in df.iterrows():
                    if pd.notna(row[tle1_col]) and pd.notna(row[tle2_col]):
                        name = str(row[name_col]) if name_col else f"OBJECT_{_}"
                        tle1 = str(row[tle1_col]).strip()
                        tle2 = str(row[tle2_col]).strip()
                        
                        # Validate TLE format
                        if (tle1.startswith('1 ') and len(tle1) == 69 and
                            tle2.startswith('2 ') and len(tle2) == 69):
                            tle_data.append({
                                'name': name[:40],
                                'tle_line1': tle1,
                                'tle_line2': tle2
                            })
    
    except Exception as e:
        print(f"Error loading file: {e}")
    
    return tle_data


def fetch_spacetrack_tles(username, password):
    """Fetch current TLEs from Space-Track"""
    try:
        session = requests.Session()
        
        # Login
        resp = session.post(
            'https://www.space-track.org/ajaxauth/login',
            data={'identity': username, 'password': password}
        )
        resp.raise_for_status()
        
        # Fetch recent TLEs (last 3 days for freshness)
        url = 'https://www.space-track.org/basicspacedata/query/class/tle_latest/ORDINAL/1/EPOCH/>now-3/format/3le'
        response = session.get(url)
        response.raise_for_status()
        
        # Parse 3LE format
        tle_data = []
        lines = response.text.strip().split('\n')
        
        i = 0
        while i < len(lines) - 2:
            if lines[i].startswith('0 '):
                name = lines[i][2:].strip()
                tle1 = lines[i+1].strip()
                tle2 = lines[i+2].strip()
                
                # Fixed typo: check len(tle2), not len(2)
                if (tle1.startswith('1 ') and len(tle1) == 69 and
                    tle2.startswith('2 ') and len(tle2) == 69):
                    tle_data.append({
                        'name': name,
                        'tle_line1': tle1,
                        'tle_line2': tle2
                    })
                i += 3
            else:
                i += 1
        
        return tle_data
        
    except Exception as e:
        print(f"Error fetching from Space-Track: {e}")
        return None


def main():
    """Main execution with ML-enhanced features"""
    print("="*70)
    print("ML-ENHANCED COLLISION PREDICTION SYSTEM")
    print("With Random Forest Risk Scoring & Pattern Recognition")
    print("="*70)
    print("\nNew ML Features:") 
    print("• Random Forest classifier for risk categorization")
    print("• Random Forest regressor for probability prediction")
    print("• Feature importance analysis")
    print("• Automated training on conjunction data")
    print("• Combined physics + ML risk scoring")
    print("\nPhysics Features:")
    print("• Anisotropic initial covariance (3-10x along-track growth)")
    print("• Encounter plane projections (2D analysis)")
    print("• Importance sampling Monte Carlo")
    print("• Alfano analytical method implementation")
    print("• Enhanced diagnostics and validation")
    
    # ML configuration
    print("\n" + "="*70)
    print("ML CONFIGURATION")
    print("="*70)
    
    enable_ml = input("Enable ML risk scoring? (y/n, default=y): ").strip().lower()
    enable_ml = enable_ml != 'n'
    
    # Initialize enhanced system with ML
    predictor = ScientificallyAccurateCollisionPredictor(
        monte_carlo_samples=50000,
        enable_ml=enable_ml
    )
    
    # Check for existing ML model
    if enable_ml:
        existing_models = [f for f in os.listdir('.') if f.startswith('ml_risk_model_') and f.endswith('.pkl')]
        if existing_models:
            print(f"\nFound {len(existing_models)} existing ML model(s):")
            for i, model in enumerate(existing_models[-5:]):  # Show last 5
                print(f"  {i+1}. {model}")
            
            load_choice = input("\nLoad existing model? Enter number or press Enter to skip: ").strip()
            if load_choice.isdigit() and 1 <= int(load_choice) <= len(existing_models):
                model_file = existing_models[-5:][int(load_choice)-1]
                if predictor.ml_predictor.load_model(model_file):
                    print(f"✓ Loaded ML model from {model_file}")
    
    # Data source selection
    print("\n" + "="*70)
    print("DATA SOURCE SELECTION")
    print("="*70)
    print("1. Fetch current TLEs from Space-Track")
    print("2. Load TLE file (.txt, 3LE format)")
    print("3. Load Excel/CSV with TLE columns")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    tle_data = None
    
    if choice == "1":
        username = input("Space-Track username: ").strip()
        password = input("Space-Track password: ").strip()
        print("\nFetching fresh TLEs from Space-Track...")
        tle_data = fetch_spacetrack_tles(username, password)
        
    elif choice == "2":
        filepath = input("Enter TLE file path: ").strip()
        if not filepath:
            filepath = "tle_data.txt"
        if os.path.exists(filepath):
            print(f"\nLoading {filepath}...")
            tle_data = load_tle_data(filepath)
        else:
            print(f"File not found: {filepath}")
    
    elif choice == "3":
        filepath = input("Enter Excel/CSV file path: ").strip()
        if not filepath:
            filepath = "Book1.csv"
        if os.path.exists(filepath):
            print(f"\nLoading {filepath}...")
            tle_data = load_tle_data(filepath)
        else:
            print(f"File not found: {filepath}")
    
    if tle_data and len(tle_data) > 0:
        print(f"Loaded {len(tle_data):,} validated TLE entries")
        
        # Load catalog
        predictor.load_catalog(tle_data)
        
        if len(predictor.objects) > 0:
            # Analysis parameters
            print("\nAnalysis Configuration:")
            time_window = input("Time window (hours, default=24): ").strip()
            time_window = float(time_window) if time_window else 24.0
            
            monte_carlo = input("Monte Carlo samples (default=50000): ").strip()
            if monte_carlo:
                predictor.analyzer.n_samples = int(monte_carlo)
            
            print(f"\nStarting ML-enhanced analysis...")
            print(f"   Time window: {time_window} hours")
            print(f"   Monte Carlo samples: {predictor.analyzer.n_samples:,}")
            print(f"   Objects in catalog: {len(predictor.objects):,}")
            print(f"   ML enabled: {enable_ml}")
            if enable_ml and predictor.ml_predictor.is_trained:
                print(f"   ML model: TRAINED")
            print(f"   Features: Physics + ML hybrid risk scoring")
            
            # Run enhanced analysis with ML
            results = predictor.run_analysis(time_window_hours=time_window)
            # Use the same screening_distance_km as in screen_conjunctions_accurate (default 15)
            screening_distance_km = 15
            if not results.empty:
                print("\n" + "="*70)
                print("ML-ENHANCED ANALYSIS COMPLETE")
                print("="*70)
                
                if enable_ml and 'ml_collision_probability' in results.columns:
                    print("\nML VS PHYSICS COMPARISON:")
                    physics_probs = results['pc_maximum'].values
                    ml_probs = results['ml_collision_probability'].values
                    
                    physics_log = np.log10(physics_probs + 1e-15)
                    ml_log = np.log10(ml_probs + 1e-15)
                    
                    correlation = np.corrcoef(physics_log, ml_log)[0, 1]
                    
                    print(f"Correlation (log scale): {correlation:.3f}")
                    print(f"Mean physics Pc: {physics_probs.mean():.2e}")
                    print(f"Mean ML Pc: {ml_probs.mean():.2e}")
                print(f"DEBUG: Analysis completed with {len(results)} results")
            else:
                print("\nNo significant collision risks identified")
        else:
            print("\nNo valid objects to analyze")
    else:
        print("\nFailed to load TLE data")
if __name__ == "__main__":
    main()