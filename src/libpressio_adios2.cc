#include "pressio_data.h"
#include "pressio_compressor.h"
#include "pressio_options.h"
#include "libpressio_ext/cpp/io.h"
#include "libpressio_ext/cpp/pressio.h"
#include "libpressio_ext/cpp/options.h"
#include "std_compat/memory.h"
#include "std_compat/optional.h"
#include "std_compat/functional.h"
#include <numeric>
#include "adios2.h"

extern "C" void libpressio_register_adios2() {
}

namespace libpressio { namespace adios2_io_ns {

class adios2_plugin : public libpressio_io_plugin {
  public:

  template <class T>
  pressio_data read_typed(adios2::IO& io, adios2::Engine& engine, std::string const& variable_name, pressio_data* buf) {
    adios2::Variable<T> var = io.InquireVariable<T>(variable_name);
    adios2::Steps steps = var.Steps();
    adios2::Dims dims = var.Shape();
    std::vector<size_t> lp_dims(dims.begin(), dims.end());
    size_t step_size = std::accumulate(lp_dims.begin(), lp_dims.end(), size_t{1}, compat::multiplies<>{});
    lp_dims.emplace_back(steps);
    pressio_data data;
    if(!buf) {
      data = pressio_data::owning(
          pressio_dtype_from_type<T>(),
          lp_dims
          );
    } else {
      data = std::move(*buf);
    }


    for (size_t i = 0; i < steps; ++i) {
      engine.BeginStep();
      adios2::Dims step_dims = var.Shape();
      if(step_dims != dims) {
        throw std::runtime_error("libpressio's ADIOS reader requires consistent dimensions per step in multi-step mode");
      }
      engine.Get(var, static_cast<T*>(data.data()) + i*step_size, adios2::Mode::Sync);
      engine.EndStep();
    }
    engine.Close();

    return data;
  }

  virtual struct pressio_data* read_impl(struct pressio_data* buf) override {
#if ADIOS2_USE_MPI
    adios2::ADIOS adios_lib(comm);
#else
    adios2::ADIOS adios_lib();
#endif

    adios2::IO io = adios_lib.DeclareIO("libpressio");
    adios2::Engine engine = io.Open(filename, adios2::Mode::Read);
    std::string var_type = io.VariableType(dsetname);

    try {
      if(var_type=="float") {
        return new pressio_data(read_typed<float>(io, engine, dsetname, buf));
      } else if (var_type == "double") {
        return new pressio_data(read_typed<double>(io, engine, dsetname, buf));
      } else if (var_type == "uint8_t") {
        return new pressio_data(read_typed<uint8_t>(io, engine, dsetname, buf));
      } else if (var_type == "uint16_t") {
        return new pressio_data(read_typed<uint16_t>(io, engine, dsetname, buf));
      } else if (var_type == "uint32_t") {
        return new pressio_data(read_typed<uint32_t>(io, engine, dsetname, buf));
      } else if (var_type == "uint64_t") {
        return new pressio_data(read_typed<uint64_t>(io, engine, dsetname, buf));
      } else if (var_type == "int8_t") {
        return new pressio_data(read_typed<int8_t>(io, engine, dsetname, buf));
      } else if (var_type == "int16_t") {
        return new pressio_data(read_typed<int16_t>(io, engine, dsetname, buf));
      } else if (var_type == "int32_t") {
        return new pressio_data(read_typed<int32_t>(io, engine, dsetname, buf));
      } else if (var_type == "int64_t") {
        return new pressio_data(read_typed<int64_t>(io, engine, dsetname, buf));
      } else {
        set_error(1, "unsupported type or missing data");
        return nullptr;
      }
    } catch(std::runtime_error const& ex) {
      set_error(1, ex.what());
      return nullptr;
    }

    return nullptr;
  }
  virtual int write_impl(struct pressio_data const*) override{
    return 0;
  }

  struct pressio_options get_options_impl() const override
  {
    struct pressio_options options;
    set(options, "io:path", filename);
    set(options, "adios2:variable_name", dsetname);
    return options;
  }

  int set_options_impl(struct pressio_options const& options) override
  {
    get(options, "io:path", &filename);
    get(options, "adios2:variable_name", &dsetname);
    return 0;
  }

  const char* version() const override { return ADIOS2_VERSION_STR; }
  int major_version() const override { return ADIOS2_VERSION_MAJOR; }
  int minor_version() const override { return ADIOS2_VERSION_MINOR; }
  int patch_version() const override { return ADIOS2_VERSION_PATCH; }

  
  struct pressio_options get_configuration_impl() const override {
    pressio_options opts;
    set(opts, "pressio:stability", "stable");
    set(opts, "pressio:thread_safe", static_cast<int32_t>(pressio_thread_safety_multiple));
    return opts;
  }

  struct pressio_options get_documentation_impl() const override {
    pressio_options opt;
    set(opt, "pressio:description", "uses ADIOS2 to read in mulitple steps of a dataset");
    return opt;
  }

  std::shared_ptr<libpressio_io_plugin> clone() override {
    return compat::make_unique<adios2_plugin>(*this);
  }
  const char* prefix() const override {
    return "adios2";
  }

  private:
    std::string filename;
    std::string dsetname;
#if ADIOS2_USE_MPI
    MPI_Comm comm = MPI_COMM_WORLD;
#endif
};

static pressio_register io_adios2_plugin(io_plugins(), "adios2", [](){ return compat::make_unique<adios2_plugin>(); });
}}

