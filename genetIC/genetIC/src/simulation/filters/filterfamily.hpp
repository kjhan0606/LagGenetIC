#ifndef IC_FILTERFAMILY_HPP
#define IC_FILTERFAMILY_HPP

#include <memory>
#include <vector>
#include <stdexcept>
#include "src/simulation/filters/filter.hpp"

namespace multilevelgrid {
  template<typename DataType, typename T>
  class MultiLevelGrid;
}

namespace filters {

  //! \class FilterFamilyBase
  //! \brief Generic class to define filters on multiple levels of grids.
  template<typename T>
  class FilterFamilyBase {
  protected:

    std::vector<std::shared_ptr<filters::Filter<T>>> filters;//!< Filters for a given level
    std::vector<std::shared_ptr<filters::Filter<T>>> complementFilters; //!< Complementary filters for a given level
    std::vector<std::shared_ptr<filters::Filter<T>>> hpFilters;//!< High pass filters for a given level
    std::vector<std::shared_ptr<filters::Filter<T>>> lpFilters;//!< Low pass filters for a given level

    //! Default constructor
    FilterFamilyBase() {

    }

  public:

    //! Constructor with known number of levels
    FilterFamilyBase(size_t maxLevels) {
      for (size_t i = 0; i < maxLevels; ++i)
        filters.push_back(std::make_shared<Filter<T>>());
    }


    //! Returns the filter on the specified level
    virtual const Filter <T> &getFilterForLevel(int level) const {
      return *(filters[level]);
    }

    //! Returns the high pass filter on the specified level
    virtual const Filter <T> &getHighPassFilterForLevel(int level) const {
      return *(hpFilters[level]);
    }

    //! Returns the low pass filter on the specified level
    virtual const Filter <T> &getLowPassFilterForLevel(int level) const {
      return *(lpFilters[level]);
    }

    //! Returns the number of levels of filter that have been defined.
    virtual size_t getMaxLevel() const {
      return filters.size();
    }

    //! Outputs debug information
    virtual void debugInfo(std::ostream &s) const {
      s << "FilterFamilyBase(";
      for (size_t i = 0; i < filters.size(); ++i) {
        s << *filters[i];
        if (i + 1 < filters.size()) s << ", ";
      }
      s << ")";

    }

    //! Add a level to this filter family.
    virtual void addLevel(T k_cut) = 0;
  };

  /*! \class FilterFamily
    \brief Implementation of default set of Fourier-space filters for combining information from fields on different levels
  */
  template<typename T>
  class FilterFamily : public FilterFamilyBase<T> {
  protected:
    using LowFilterType = filters::LowPassFermiFilter<T>; // this could be templatised to allow more flexibility
    using HighFilterType = filters::ComplementaryCovarianceFilterAdaptor<LowFilterType>;
  public:

    template<typename S>
    explicit FilterFamily(const multilevelgrid::MultiLevelGridBase<T, S> &fromContext) {
      if(fromContext.getLevelsAreCombined()) {
        // Output fields have been generated already...
        // Regard levels as independent. Strictly the filters should now be spatial, i.e. each grid
        // is used in its own domain of validity, but we assume we are only interested in the highest
        // resolution area (since in practice the only place the filters will now be used is in
        // checking modification values).
        auto nullfilter = std::make_shared<NullFilter<T>>();
        auto identityfilter = std::make_shared<Filter<T>>();
        for(size_t level=0; level < fromContext.getNumLevels() - 1; ++level) {
          this->filters.push_back(nullfilter);
          this->lpFilters.push_back(nullfilter);
          this->hpFilters.push_back(nullfilter);
        }
        this->filters.push_back(identityfilter);
        this->lpFilters.push_back(identityfilter);
        this->hpFilters.push_back(identityfilter);

      } else {
        // Build the filter ladder over distinct scale *rungs* rather than over storage levels. For a
        // conventional linear zoom stack each level is its own rung, so this reproduces the original
        // behaviour exactly. With disjoint sibling zoom boxes (multi-void) several levels share a rung,
        // and we then assign each level the filter of its rung so that siblings are filtered identically.
        size_t numRungs = fromContext.getNumRungs();

        this->filters.push_back(std::make_shared<Filter<T>>());
        this->lpFilters.push_back(std::make_shared<Filter<T>>());
        this->hpFilters.push_back(std::make_shared<Filter<T>>());

        for (size_t rung = 0; rung + 1 < numRungs; ++rung) {
          // Any level at this rung shares the same cellSize, hence the same pixel-scale cut.
          const grids::Grid<T> &gridAtRung(representativeGridForRung(fromContext, rung));

          T k_pixel = ((T) gridAtRung.size) * gridAtRung.getFourierKmin();
          T k_cut = FRACTIONAL_K_SPLIT * k_pixel;
          this->addLevel(k_cut);
        }

        // At this point filters/lpFilters/hpFilters are indexed by rung (size numRungs). If the stack is a
        // genuine tree (more levels than rungs) remap them so they are indexed by storage level, with each
        // level taking its rung's filter. For a linear stack (numRungs == numLevels) this is a no-op.
        if (numRungs != fromContext.getNumLevels())
          expandRungFiltersToLevels(fromContext);
      }
    }

  protected:
    //! Returns a grid representative of the given rung (the first stored level whose rung matches).
    template<typename S>
    static const grids::Grid<T> &representativeGridForRung(
        const multilevelgrid::MultiLevelGridBase<T, S> &context, size_t rung) {
      for (size_t level = 0; level < context.getNumLevels(); ++level)
        if (context.getRungForLevel(level) == rung)
          return context.getGridForLevel(level);
      throw std::runtime_error("FilterFamily: no grid found for requested rung");
    }

    //! Re-index the rung-keyed filter vectors so they are keyed by storage level (each level -> its rung's filter).
    template<typename S>
    void expandRungFiltersToLevels(const multilevelgrid::MultiLevelGridBase<T, S> &context) {
      auto rungFilters = this->filters;
      auto rungLp = this->lpFilters;
      auto rungHp = this->hpFilters;
      this->filters.clear();
      this->lpFilters.clear();
      this->hpFilters.clear();
      for (size_t level = 0; level < context.getNumLevels(); ++level) {
        size_t rung = context.getRungForLevel(level);
        this->filters.push_back(rungFilters[rung]);
        this->lpFilters.push_back(rungLp[rung]);
        this->hpFilters.push_back(rungHp[rung]);
      }
    }

  public:

    //! Adds relevant filters to the next level to be defined, based on what was used on the previous level
    void addLevel(T k_cut) override {
      std::shared_ptr<Filter<T>> filt = this->filters.back();
      this->filters.pop_back();
      this->lpFilters.pop_back();

      // the low-pass filtered version of the field on what used to be the finest level:
      this->filters.push_back(((*filt) * LowFilterType(k_cut)).clone());

      this->complementFilters.push_back(((*filt) * HighFilterType(k_cut)).clone());

      // the low-pass filter to be used when copying from what used to be the finest level to the new finest level:
      this->lpFilters.push_back(LowFilterType(k_cut).clone());

      // the new finest level filter:
      this->filters.push_back(((*filt) * HighFilterType(k_cut)).clone());

      // the high-pass filter for removing unwanted low-k information from the new fine level:
      this->hpFilters.push_back(HighFilterType(k_cut).clone());

      // if this ends up being the last level, there is no low-pass filter ever applied
      this->lpFilters.push_back(Filter<T>().clone());

    }

  };


  //! Output debug information about a filter to a stream
  template<typename T>
  std::ostream &operator<<(std::ostream &s, const Filter <T> &f) {
    f.debugInfo(s);
    return s;
  }

  //! Output debug information about a family of filters to a stream
  template<typename T>
  std::ostream &operator<<(std::ostream &s, const FilterFamilyBase<T> &f) {
    f.debugInfo(s);
    return s;
  }
}
#endif
