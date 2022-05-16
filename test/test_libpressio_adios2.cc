#include "gtest/gtest.h"
#include <adios2.h>
#include <libpressio_ext/cpp/pressio.h>
#include <iostream>

constexpr const char* filepath = CMAKE_PROJECT_SOURCE_DIR "/test/ADIOS2ADIOS1WriteADIOS1Read1D8.bp";

template <class T>
void print_var_info(adios2::IO& io, adios2::Engine& engine, std::string var_name) {
  adios2::Variable<T> var = io.InquireVariable<T>(var_name);
  adios2::Dims dims = var.Shape();
  adios2::Steps steps = var.Steps();
  //std::cout << "{";
  //for (auto const& i : dims) {
  //  std::cout << i << ", ";
  //}
  //std::cout << "}";
  //std::cout << "steps=" << steps << std::endl;
}

TEST(libpressio_adios2, experiments) {
  adios2::ADIOS library;
  adios2::IO io = library.DeclareIO("foobar");
  adios2::Engine engine = io.Open(filepath, adios2::Mode::Read);
  auto vars = io.AvailableVariables();
  for (auto const& i : vars) {
    auto var_type = io.VariableType(i.first);
    //std::cout << i.first << ' ' << var_type << std::endl;
    if(var_type=="float") {
      print_var_info<float>(io, engine, i.first);
    } else if (var_type == "double") {
      print_var_info<double>(io, engine, i.first);
    } else if (var_type == "uint8_t") {
      print_var_info<uint8_t>(io, engine, i.first);
    } else if (var_type == "uint16_t") {
      print_var_info<uint16_t>(io, engine, i.first);
    } else if (var_type == "uint32_t") {
      print_var_info<uint32_t>(io, engine, i.first);
    } else if (var_type == "uint64_t") {
      print_var_info<uint64_t>(io, engine, i.first);
    } else if (var_type == "int8_t") {
      print_var_info<int8_t>(io, engine, i.first);
    } else if (var_type == "int16_t") {
      print_var_info<int16_t>(io, engine, i.first);
    } else if (var_type == "int32_t") {
      print_var_info<int32_t>(io, engine, i.first);
    } else if (var_type == "int64_t") {
      print_var_info<int64_t>(io, engine, i.first);
    }
    for (auto const& param : i.second) {
      //std::cout << '\t' << param.first << ' ' << param.second << std::endl;
    }
  }
}

TEST(libpressio_adios2, integration1) {
  pressio library;
  pressio_io io = library.get_io("adios2");
  io->set_options({
      {"io:path", filepath},
      {"adios2:variable_name", "r64"}
      });
  pressio_data* read = io->read(nullptr);

  GTEST_ASSERT_NE(read, nullptr);
  GTEST_ASSERT_EQ(read->dtype(), pressio_double_dtype);
  std::vector<size_t> expected_dims{288,3};
  GTEST_ASSERT_EQ(read->dimensions(), expected_dims);
}

int main(int argc, char *argv[])
{
  int rank, size, disable_printers=1;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  ::testing::InitGoogleTest(&argc, argv);

  if(rank == 0){
    int opt;
    while((opt = getopt(argc, argv, "p")) != -1) {
        switch(opt) {
          case 'p':
          disable_printers = 0;
          break;
        default:
          break;
        }
    }
  }
  MPI_Bcast(&disable_printers, 1, MPI_INT, 0, MPI_COMM_WORLD);

  //disable printers for non-root process
  if(rank != 0 and disable_printers) {
    auto&& listeners = ::testing::UnitTest::GetInstance()->listeners();
    delete listeners.Release(listeners.default_result_printer());
  }

  int result = RUN_ALL_TESTS();

  int all_result=0;
  MPI_Allreduce(&result, &all_result, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if(rank == 0 && all_result) std::cerr << "one or more tests failed on another process, please check them" << std::endl;
  MPI_Finalize();

  return all_result;
}
