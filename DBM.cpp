#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <fstream>

using namespace std;

class Cell
{
public:
    int i, j;
    double phi;
    vector<pair<int, int>> neighbors;

    Cell(int i, int j, double potential = 0.0) : i(i), j(j), phi(potential)
    {
        for (int di = -1; di <= 1; ++di)
        {
            for (int dj = -1; dj <= 1; ++dj)
            {
                if (di == 0 && dj == 0)
                    continue;
                neighbors.push_back({i + di, j + dj});
            }
        }
    }

    pair<int, int> get_coords()
    {
        return {i, j};
    }
};

class DielectricBreakdownModel
{
    double eta, R1;
    int dim;
    vector<vector<int>> grid;
    vector<Cell> pattern;
    vector<Cell> candidates;
    vector<Cell> environment;
    bool hit_edge = false;
    mt19937 gen;

public:
    DielectricBreakdownModel(double eta = 5.0, int dim = 100, double h = 1.0, pair<int, int> seed = {-1, -1})
        : eta(eta), R1(h / 2), dim(dim), gen(random_device{}())
    {

        grid.resize(2 * dim + 1, vector<int>(2 * dim + 1, 0));
        if (seed.first == -1)
            seed = {dim, dim};

        Cell seedCell(seed.first, seed.second);
        add_cell(seedCell);
    }

    void grow_pattern()
    {
        Cell cell = choose_candidate();
        add_cell(cell);
        update_potentials(cell);
    }

    Cell choose_candidate()
    {
        vector<double> phi_values;
        for (const auto &c : candidates)
        {
            phi_values.push_back(c.phi);
        }
        double max_phi = *max_element(phi_values.begin(), phi_values.end());
        double min_phi = *min_element(phi_values.begin(), phi_values.end());

        vector<double> phi_eta;
        double phi_eta_sum = 0;
        for (auto phi : phi_values)
        {
            double value = pow((phi - min_phi) / (max_phi - min_phi), eta);
            phi_eta.push_back(value);
            phi_eta_sum += value;
        }

        discrete_distribution<> dist(phi_eta.begin(), phi_eta.end());
        int idx = dist(gen);

        Cell chosen = candidates[idx];
        candidates.erase(candidates.begin() + idx);
        return chosen;
    }

    int simulate(int steps = 0)
    {
        if (steps == 0)
        {
            while (!hit_edge)
            {
                grow_pattern();
                ++steps;
                if (steps % 200 == 0)
                {
                    save_grid_to_file("frames/" + to_string(steps) + ".csv");
                }
            }
        }
        else
        {
            for (int i = 0; i < steps; ++i)
            {
                grow_pattern();
            }
        }
        return steps;
    }

    void add_cell(const Cell &cell)
    {
        grid[cell.i][cell.j] = 1;
        pattern.push_back(cell);

        auto neighbors = cell.neighbors;
        neighbors.erase(std::remove_if(neighbors.begin(), neighbors.end(), [this](const pair<int, int> &p)
                                       { return p.first < 0 || p.first >= 2 * dim + 1 || p.second < 0 || p.second >= 2 * dim + 1; }),
                        neighbors.end());

        if (neighbors.size() < 8)
        {
            hit_edge = true;
        }

        for (const auto &[ni, nj] : neighbors)
        {
            if (grid[ni][nj] != 0)
                continue;
            grid[ni][nj] = -1;
            candidates.emplace_back(ni, nj, calculate_spawn_potential(ni, nj));
        }
    }

    double calculate_spawn_potential(int i, int j)
    {
        double neighbor_potential = 0.0;
        for (const auto &p : pattern)
        {
            double dist = sqrt(pow(i - p.i, 2) + pow(j - p.j, 2));
            neighbor_potential += 1 - R1 / dist;
        }

        double environment_potential = 0.0;
        for (const auto &c : environment)
        {
            double dist = sqrt(pow(i - c.i, 2) + pow(j - c.j, 2));
            environment_potential += c.phi / dist;
        }

        return neighbor_potential + environment_potential;
    }

    void update_potentials(const Cell &cell)
    {
        for (auto &c : candidates)
        {
            double dist = sqrt(pow(c.i - cell.i, 2) + pow(c.j - cell.j, 2));
            c.phi += 1 - R1 / dist;
        }
    }

    void save_grid_to_file(const string &filename)
    {
        ofstream file(filename);
        if (file.is_open())
        {
            for (const auto &row : grid)
            {
                for (size_t j = 0; j < row.size(); ++j)
                {
                    file << row[j];
                    if (j < row.size() - 1)
                        file << ",";
                }
                file << "\n";
            }
            file.close();
        }
        else
        {
            cerr << "Unable to open file: " << filename << endl;
        }
    }
};

int main()
{
    DielectricBreakdownModel dbm(5.0, 1000);
    int steps = dbm.simulate();
    cout << "Simulation finished after " << steps << " steps.\n";

    return 0;
}
