#ifndef IC_MASK_HPP
#define IC_MASK_HPP

#include "src/simulation/multilevelgrid/multilevelgrid.hpp"

namespace multilevelgrid {

  //! Abstract class to generate masks through the multilevel hierarchy
  template<typename DataType, typename T=tools::datatypes::strip_complex<DataType>>
  class AbstractBaseMask {
  public:

    //! Constructor from the specified multi-level context.
    explicit AbstractBaseMask(MultiLevelGrid <DataType> *multilevelgrid_) :
      multilevelgrid(multilevelgrid_), flaggedIdsAtEachLevel(multilevelgrid->getNumLevels()) {}

    /*! \brief Returns 1 if the specified cell is in the mask, 0 otherwise.

        \param level - level of cell to check.
        \param cellindex - index of cell to check on the specified level.
    */
    virtual T const isInMask(size_t level, size_t cellindex) = 0;

    //! Currently unimplemented.
    virtual void ensureFlaggedVolumeIsContinuous() = 0;

    //! Processes the information in the multi-level context and uses it to create a graphic mask.
    virtual void calculateMask() = 0;

  protected:
    MultiLevelGrid <DataType> *multilevelgrid; //!< Pointer to the multi-level context object.
    std::vector<std::vector<size_t>> flaggedIdsAtEachLevel; //!< Vector whose elements are vectors of the ids flagged at each level of the mask.

    //! Calculates the flagged cells on all levels.
    virtual void generateFlagsHierarchy() = 0;

  };

  /*! \class GraficMask
      \brief Extend the concept of mask for grafic outputs.

      The masks must be carried through the entire grafic hierarchy, including virtual intermediate grids.
      Useful for generating ic_refmap/ic_pvar files for RAMSES for example
  */
  template<typename DataType, typename T=tools::datatypes::strip_complex<DataType>>
  class GraficMask : public AbstractBaseMask<DataType, T> {

  private:
    const std::vector<std::vector<size_t>> &inputzoomParticlesAsMask;
    int deepestLevelWithMaskedCells = -1;

  public:
    /*! \brief Constructor, requires a pointer to the multi-level context and a reference to a vector of mask vectors for each level
        \param multilevelgrid_ - pointer to the multi-level context object.
        \param input_mask - vector of vectors, where each vector is a mask for a given level.
    */
    explicit GraficMask(MultiLevelGrid <DataType> *multilevelgrid_,
                        std::vector<std::vector<size_t>> &input_mask) :
      AbstractBaseMask<DataType, T>(multilevelgrid_), inputzoomParticlesAsMask(input_mask) {
      assert(inputzoomParticlesAsMask.size() <= this->flaggedIdsAtEachLevel.size());
    };

    /*!
     * @return 1.0 if cell with index is in the mask in this level, 0.0 else.
     */
    T const isInMask(size_t level, size_t cellindex) override {

      if (this->flaggedIdsAtEachLevel[level].size() == 0) {
        // No flagged cells at all on this level, refine everywhere
        return T(1.0);
      } else if (isMasked(cellindex, level)) {
        // Cell is among flagged cells, refine it
        return T(1.0);
      }
      return T(0.0);

    }

    //! Processes the information in the multi-level context and uses it to create a graphic mask
    void calculateMask() override {
      identifyLevelsOfInputMask();
      generateFlagsHierarchy();
      ensureFlaggedVolumeIsContinuous();
    }

    //! Currently unimplemented.
    void ensureFlaggedVolumeIsContinuous() override {
      //TODO Add Lagrangian volume definition if it bothers you
    }


  protected:
    //! Returns the nearest real (non-virtual) ancestor of the given level by walking parent pointers.
    size_t nearestRealAncestor(size_t level) {
      size_t p = this->multilevelgrid->getParentLevel(level);
      while (!this->multilevelgrid->isBaseLevel(p) &&
             this->multilevelgrid->getGridForLevel(p).isUpsampledOrDownsampled()) {
        p = this->multilevelgrid->getParentLevel(p);
      }
      return p;
    }

    //! Returns true if the level is a real grid with no finer children (the finest box of its branch).
    bool isRealLeaf(size_t level) {
      bool real = !this->multilevelgrid->getGridForLevel(level).isUpsampledOrDownsampled();
      return real && this->multilevelgrid->getChildLevels(level).empty();
    }

    /*! \brief Match each input mask to the real grid whose region it describes.
     *
     * Input mask k holds the parent-grid cells that were flagged to open the (k+1)-th real zoom grid.
     * Enumerating real grids in storage order, the k-th input mask therefore belongs to the nearest
     * real ancestor of the (k+1)-th real grid. For disjoint sibling boxes (multi-void) several zooms
     * share the base as parent, so their masks are merged onto the base level rather than assigned to
     * successive coarse levels (which is what the old linear-hierarchy logic assumed).
     */
    void identifyLevelsOfInputMask() {
      std::vector<size_t> realLevels;
      for (size_t lvl = 0; lvl < this->multilevelgrid->getNumLevels(); ++lvl) {
        if (!this->multilevelgrid->getGridForLevel(lvl).isUpsampledOrDownsampled())
          realLevels.push_back(lvl);
      }

      for (size_t k = 0; k < this->inputzoomParticlesAsMask.size(); ++k) {
        if (this->inputzoomParticlesAsMask[k].empty())
          continue;
        if (k + 1 >= realLevels.size()) {
          logging::entry(logging::level::warning)
            << "Grafic mask: more input masks than zoom grids; ignoring extras" << std::endl;
          break;
        }

        size_t parentReal = nearestRealAncestor(realLevels[k + 1]);
        auto &dst = this->flaggedIdsAtEachLevel[parentReal];
        dst.insert(dst.end(),
                   this->inputzoomParticlesAsMask[k].begin(),
                   this->inputzoomParticlesAsMask[k].end());
        tools::sortAndEraseDuplicate(dst);

        if (int(parentReal) > this->deepestLevelWithMaskedCells)
          this->deepestLevelWithMaskedCells = int(parentReal);
      }
    }

    //! \brief Calculates the flagged cells on all levels by propagating each real grid's mask down the tree.
    /*!
     * Real grids carrying an input mask keep it. Virtual intermediate grids inherit their parent's mask
     * (projected by spatial location, so disjoint sibling branches stay separated). Real leaf grids (the
     * finest box of a branch) are left empty, which isInMask interprets as "refine everywhere".
     */
    void generateFlagsHierarchy() override {

      if (this->deepestLevelWithMaskedCells < 0) {
        logging::entry(logging::level::warning) << "WARNING No zoom regions were ever opened. Grafic mask will not be generated in this case"
                  << std::endl;
        return;
      }

      // Ascending storage order guarantees a level's parent is processed before the level itself.
      for (size_t level = 0; level < this->multilevelgrid->getNumLevels(); ++level) {
        if (this->multilevelgrid->isBaseLevel(level))
          continue;                                       // base keeps its merged input mask
        if (!this->flaggedIdsAtEachLevel[level].empty())
          continue;                                       // real grid with its own input mask
        if (isRealLeaf(level))
          continue;                                       // finest box of a branch: refine everywhere

        size_t parent = this->multilevelgrid->getParentLevel(level);
        const auto &grid = this->multilevelgrid->getGridForLevel(level);
        for (size_t i = 0; i < grid.size3; ++i) {
          size_t parentIndex = this->multilevelgrid->getIndexOfCellOnOtherLevel(level, parent, i);
          if (this->isMasked(parentIndex, parent))
            this->flaggedIdsAtEachLevel[level].push_back(i);
        }
        sortAndEraseDuplicate(level);
      }
    }

    //! Sorts the flags on the specified level and erases any duplicates.
    void sortAndEraseDuplicate(size_t level) {
      tools::sortAndEraseDuplicate(this->flaggedIdsAtEachLevel[level]);
    }

    /*! \brief Returns true if the cell at the specified level is flagged for inclusion in the mask
        \param id - cell id to check
        \param level - level the id corresponds to
    */
    bool isMasked(size_t id, size_t level) {
      return std::binary_search(this->flaggedIdsAtEachLevel[level].begin(),
                                this->flaggedIdsAtEachLevel[level].end(), id);
    }

  public:
    //! Generate a full multilevel field storing the mask information.
    /*! Increases memory requirement for the code but useful for debugging.
     */
    std::shared_ptr<fields::MultiLevelField<DataType>> convertToField() {
      std::vector<std::shared_ptr<fields::Field<DataType, T>>> fields;

      // Field full of zeros
      for (size_t level = 0; level < this->multilevelgrid->getNumLevels(); ++level) {
        fields.push_back(std::make_shared<fields::Field<DataType, T>>(
          this->multilevelgrid->getGridForLevel(level)));
      }

      auto maskfield = std::make_shared<fields::MultiLevelField<DataType>>(*(this->multilevelgrid), fields);

      // Field with mask information
      for (size_t level = 0; level < this->multilevelgrid->getNumLevels(); ++level) {
        for (size_t i_z = 0; i_z < this->multilevelgrid->getGridForLevel(level).size; ++i_z) {
          for (size_t i_y = 0; i_y < this->multilevelgrid->getGridForLevel(level).size; ++i_y) {
            for (size_t i_x = 0; i_x < this->multilevelgrid->getGridForLevel(level).size; ++i_x) {

              // These two indices can be different if some virtual grid are used in the context, e.g. centered.
              // In all other cases, they will be equal.
              size_t i = size_t(i_x * this->multilevelgrid->getGridForLevel(level).size + i_y)
                         * this->multilevelgrid->getGridForLevel(level).size + i_z;
              size_t virtual_i = this->multilevelgrid->getGridForLevel(level).getIndexFromCoordinateNoWrap(i_x, i_y,
                                                                                                              i_z);

              maskfield->getFieldForLevel(level).getDataVector()[i] = isInMask(level, virtual_i);

            }
          }
        }
      }
      return maskfield;
    }
  };
}


#endif
