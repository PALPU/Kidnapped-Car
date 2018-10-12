
#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	
static int Num_Particles=100;
void ParticleFilter::init(double x, double y, double theta, double std[]) 
{
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    std::normal_distribution<double> n_x_(x, std[0]);
	std::normal_distribution<double> n_y_(y, std[1]);
	std::normal_distribution<double> n_theta_(theta, std[2]);
	std::default_random_engine gen;
    //Resizing the  weight and vectors of the particles
	num_particles = Num_Particles;
	particles.resize(num_particles);

	//Initializing the Particles
	for(auto& Op: particles)
	{
       Op.y = n_y_(gen);
	   Op.x = n_x_(gen);
	   Op.weight = 1;
	   Op.theta = n_theta_(gen);
	   
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
    std::default_random_engine gen;
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
    std::normal_distribution<double> N_x(0, std_pos[0]);
	std::normal_distribution<double> N_y(0, std_pos[1]);
	std::normal_distribution<double> N_theta(0, std_pos[2]);
	for (auto& Op: particles)
	{
	  // New State Calculations...
	   if( fabs(yaw_rate) < 0.0001)
 	   {
            Op.x += velocity * delta_t * cos(Op.theta);
            Op.y += velocity * delta_t * sin(Op.theta);
        }
	    else
    	{
            Op.x += velocity / yaw_rate * (sin( Op.theta + yaw_rate*delta_t) - sin(Op.theta) );
            Op.y += velocity / yaw_rate * (cos( Op.theta) - cos(Op.theta + yaw_rate*delta_t ) );
            Op.theta += yaw_rate * delta_t;
        }
		
	  //Predicted Particles and Sensor Noise...
	    Op.x +=N_x(gen);
	    Op.y +=N_y(gen);
	    Op.theta +=N_theta(gen);
	 }	

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) 
{
    double min_dist;
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
    for (auto &Obs_o: observations)
    {

        //Initialising Minimum to Maximum Distance Possible
        min_dist = std::numeric_limits<float>::max();
        for (const auto &Pred_o: predicted)
        {
        //Distance b/w current and predicted landmark...
            double current_dist = dist(Obs_o.x, Obs_o.y, Pred_o.x, Pred_o.y); 

        //Predicting the nearest landmark to current observed landmark...
            if(current_dist < min_dist)
            {
                min_dist = current_dist;
                //Setting Observation's Id as per nearest predicted landmark's id
                Obs_o.id = Pred_o.id;
            }
        }

   }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) 
{
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
    for (auto& Op: particles)
	{	
	    //Update Weight
	    Op.weight = 1.0;
		
        //Vector to store Landmarks Location Predictions within sensor range...
        vector<LandmarkObs> predictions;

        //Map Landmarks...
        for(const auto &lml: map_landmarks.landmark_list)
        {
            double distance = dist(Op.x, Op.y, lml.x_f, lml.y_f);
            
            //To Consider the Landmarks which are only within sensor range of particle...
            if(distance < sensor_range)
            {
                //Adding prediction to vector...
                predictions.push_back(LandmarkObs{lml.id_i, lml.x_f, lml.y_f});
            }
        }
        //Creating the copy of list of observations transformed from vehicle coordinates to map coordinates...
        vector<LandmarkObs> transformed_obs;
        double cos_value = cos(Op.theta);
        double sin_value = sin(Op.theta);

        for (const auto &Obs_o: observations)
        {
            LandmarkObs t;
            t.x = cos_value * Obs_o.x - sin(Op.theta) * Obs_o.y + Op.x;
            t.y = sin_value * Obs_o.x + cos(Op.theta) * Obs_o.y + Op.y;
            transformed_obs.push_back(t);
		}

        //To Perform dataAssociation for the predicted and transformed observations of the current particles...
        dataAssociation(predictions, transformed_obs);

        //Computing  Weight
        for (const auto& trans_obs: transformed_obs)
        {
            //Placeholders for Observations and Associated Predicted Coordinates
            Map::single_landmark_s landmark = map_landmarks.landmark_list.at(trans_obs.id-1);
            double trans_obs_x = pow(trans_obs.x - landmark.x_f, 2) / (2 * pow(std_landmark[0], 2));
            double trans_obs_y = pow(trans_obs.y - landmark.y_f, 2) / (2 * pow(std_landmark[1], 2));
                
            //Calculation of weight of current observation using Multivariate Guassian
            double w = exp(-(trans_obs_x + trans_obs_y)) / (2 * M_PI * std_landmark[0] * std_landmark[1]);

            //Current Observation Weight * Total Observable Weight
            Op.weight *= w;

        }
      	weights.push_back(Op.weight);
   }

}

void ParticleFilter::resample() 
{
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    std::random_device rd;
	std::mt19937 gen(rd());
	std::discrete_distribution<> dist(weights.begin(), weights.end());

	//creating vector for the new particles
	vector<Particle> new_particles;
	new_particles.resize(num_particles);

	//Fetching Current Weights and putting in it the vector new_particles

	for (int j = 0; j < num_particles; j++)
    {
	    int ids = dist(gen);
	    new_particles[j] = particles[ids];
	}
	particles = new_particles;
	weights.clear();
	
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates
    particle.associations.clear();
    particle.sense_x.clear();
    particle.sense_y.clear();
    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
