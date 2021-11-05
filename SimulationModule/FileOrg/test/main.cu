#include <string>
#include <stdio.h>

void print_file( std::string file_path){
    printf("%s\n", file_path.c_str() );
    return;
}

void print_path( std::string dir ){
    print_file( dir+"tmp.dat" );
    return;
}


int main(){
    print_path( "./" );

    return 0;
}
