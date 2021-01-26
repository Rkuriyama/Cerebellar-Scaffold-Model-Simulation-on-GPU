#include <iostream>
#include <fstream>
#include <string>

#include <random>


int main( int argc, char **argv){

    std::ofstream ofs("purkinje_to_dcn.dat");

    std::string str;

    std::random_device rnd;
    std::mt19937 mt( rnd() );
    std::uniform_real_distribution<> randR(0.0, 1.0);

    int count[2] = {0};

    std::ifstream ifs0("PC0.dat");
    while( std::getline(ifs0, str) ){
        int pc_id = std::stoi(str);
        for(int dcn_id = 0; dcn_id < 6; dcn_id++){
            if( randR(mt) < 4./6. ){
                 ofs << pc_id << "\t" << dcn_id << std::endl;
                 count[0]++;
            }
        }
    }
    ifs0.close();

    std::ifstream ifs1("PC1.dat");
    while( std::getline(ifs1, str) ){
        int pc_id = std::stoi(str);
        for(int dcn_id = 6; dcn_id < 12; dcn_id++){
            if( randR(mt) < 4./6. ){
                ofs << pc_id << "\t" << dcn_id << std::endl;
                count[1]++;
            }
        }
    }
    ifs1.close();

    std::cout << count[0] << ", " << count[1] << std::endl;

    return 0;
}
