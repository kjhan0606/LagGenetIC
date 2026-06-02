#ifndef IC_GRAFIC_UNION_HPP
#define IC_GRAFIC_UNION_HPP

#include <cmath>
#include <vector>
#include <limits>
#include <algorithm>
#include "src/simulation/multilevelgrid/multilevelgrid.hpp"
#include "src/simulation/grid/virtualgrid.hpp"

namespace multilevelgrid {

  /*! \brief Collapse a multi-void sibling-tree context into a linear "union" context for grafic output.
   *
   * RAMSES/grafic expects exactly one rectangular grid per refinement level (level_min..level_max written in
   * increasing resolution). A multi-void run, however, produces several disjoint sibling zoom boxes that share
   * the same rung (resolution). Writing them naively makes every sibling at a given resolution land in the same
   * output directory (named by effective resolution), so siblings overwrite each other.
   *
   * Following MUSIC's region_multibox/output_grafic2 approach, we emit ONE grid per rung whose extent is the
   * union bounding box enclosing all that rung's siblings. Inside each sibling window the union grid evaluates to
   * that sibling's real zoom field; elsewhere it interpolates the coarser parent. Disjoint voids are then
   * distinguished purely by the refinement mask (ic_refmap), exactly as RAMSES expects.
   *
   * The union grid is assembled from existing virtual-grid primitives: a SectionOfGrid of the base covering the
   * union AABB provides the interpolated background, and the sibling windows are layered on with nested
   * ResolutionMatchingGrids (the first sibling of each rung steps up in resolution from the coarser parent;
   * subsequent siblings are layered at equal resolution, i.e. a unit-ratio RMG). Because every rung now holds a
   * single grid, the resulting context is a plain linear stack and the well-tested linear grafic path applies.
   *
   * \param tree       the (tree) grafic context produced by copyContextWithCenteredIntermediate
   * \param input_mask per-void flagged base-grid cell ids (one vector per disjoint void)
   * \param outLinear  receives the collapsed linear union context (cleared first)
   * \param outFlags   receives the per-level refinement mask (flagged cell ids on each union grid)
   */
  //! Copy every level of src into dst (cleared first), preserving the parent-pointer topology.
  template<typename DataType>
  void adoptLevels(MultiLevelGrid<DataType> &src, MultiLevelGrid<DataType> &dst) {
    auto grids = src.getAllGrids();
    dst.clear();
    for (size_t level = 0; level < grids.size(); ++level)
      dst.addLevel(grids[level], src.getParentLevel(level));
  }

  template<typename DataType, typename T=tools::datatypes::strip_complex<DataType>>
  void buildGraficUnionContext(MultiLevelGrid<DataType> &tree,
                               const std::vector<std::vector<size_t>> &input_mask,
                               MultiLevelGrid<DataType> &outLinear,
                               std::vector<std::vector<size_t>> &outFlags) {

    using grids::Grid;
    using GridPtr = std::shared_ptr<Grid<T>>;

    auto grids = tree.getAllGrids();
    const size_t nL = tree.getNumLevels();
    const size_t numRungs = tree.getNumRungs();

    GridPtr baseGrid = grids[0];

    // --- Union axis-aligned bounding box over every sibling, expressed in base-grid cells ---
    constexpr int INTMAX = std::numeric_limits<int>::max();
    Coordinate<int> aabbLo(INTMAX, INTMAX, INTMAX);
    Coordinate<int> aabbHi(-INTMAX, -INTMAX, -INTMAX);

    // Lower corners floor and upper corners ceil to base cells, so the AABB strictly encloses every sibling
    // even when a centered-intermediate grid starts on a half-base-cell. A baseSection that merely rounded to
    // the nearest base cell could land above a sibling's true lower corner, pushing it outside the background
    // grid and tripping ResolutionMatchingGrid's containsPoint assertion.
    auto floorBaseCell = [&](T worldComponent, T baseLowerComponent) {
      return int(std::floor((worldComponent - baseLowerComponent) / baseGrid->cellSize));
    };
    auto ceilBaseCell = [&](T worldComponent, T baseLowerComponent) {
      return int(std::ceil((worldComponent - baseLowerComponent) / baseGrid->cellSize));
    };

    for (size_t level = 1; level < nL; ++level) {
      const Grid<T> &g = *grids[level];
      Coordinate<int> lo(floorBaseCell(g.offsetLower.x, baseGrid->offsetLower.x),
                         floorBaseCell(g.offsetLower.y, baseGrid->offsetLower.y),
                         floorBaseCell(g.offsetLower.z, baseGrid->offsetLower.z));
      Coordinate<int> hi(ceilBaseCell(g.offsetLower.x + g.size * g.cellSize, baseGrid->offsetLower.x),
                         ceilBaseCell(g.offsetLower.y + g.size * g.cellSize, baseGrid->offsetLower.y),
                         ceilBaseCell(g.offsetLower.z + g.size * g.cellSize, baseGrid->offsetLower.z));
      aabbLo.x = std::min(aabbLo.x, lo.x);
      aabbLo.y = std::min(aabbLo.y, lo.y);
      aabbLo.z = std::min(aabbLo.z, lo.z);
      aabbHi.x = std::max(aabbHi.x, hi.x);
      aabbHi.y = std::max(aabbHi.y, hi.y);
      aabbHi.z = std::max(aabbHi.z, hi.z);
    }

    // ResolutionMatchingGrid assumes the hi-res window lies strictly interior to its low-res background: it
    // evaluates getCentroidFromCoordinate at the window's *exclusive* upper corner, which asserts unless that
    // corner is inside the grid. A sibling flush against the tight union AABB would sit exactly on the edge, so
    // grow the AABB by a one-base-cell rim on every side. The rim is interpolated background (never flagged in
    // the footprint), so it adds no spurious refinement.
    constexpr int kUnionPad = 1;
    aabbLo -= Coordinate<int>(kUnionPad, kUnionPad, kUnionPad);
    aabbHi += Coordinate<int>(kUnionPad, kUnionPad, kUnionPad);

    // Pad the union region to a cube (SectionOfGrid is cubic). Background cells outside the voids are masked out.
    int cube = std::max({aabbHi.x - aabbLo.x, aabbHi.y - aabbLo.y, aabbHi.z - aabbLo.z});
    GridPtr baseSection = std::make_shared<grids::SectionOfGrid<T>>(
      baseGrid, aabbLo.x, aabbLo.y, aabbLo.z, size_t(cube));

    // --- Build one union grid per rung, layering siblings with nested ResolutionMatchingGrids ---
    outLinear.clear();
    outLinear.addLevel(baseGrid);                 // rung 0: the full base grid (output level_min)

    GridPtr loResForRung = baseSection;           // coarser background for the next rung
    for (size_t rung = 1; rung < numRungs; ++rung) {
      GridPtr u = loResForRung;
      for (size_t level = 1; level < nL; ++level) {
        if (tree.getRungForLevel(level) != rung)
          continue;
        u = std::make_shared<grids::ResolutionMatchingGrid<T>>(grids[level], u);
      }
      outLinear.addLevel(u);                      // linear: parent is the previous rung
      loResForRung = u;
    }

    // --- Merged refinement footprint (union of all per-void base-cell masks) ---
    std::vector<size_t> mergedFootprint;
    for (const auto &voidMask : input_mask)
      mergedFootprint.insert(mergedFootprint.end(), voidMask.begin(), voidMask.end());
    tools::sortAndEraseDuplicate(mergedFootprint);

    auto inFootprint = [&](size_t baseId) {
      return std::binary_search(mergedFootprint.begin(), mergedFootprint.end(), baseId);
    };

    // --- Per-level masks: project each union cell down to the base and test the merged footprint ---
    size_t outLevels = outLinear.getNumLevels();
    outFlags.assign(outLevels, {});
    outFlags[0] = mergedFootprint;                // base level keeps the merged footprint directly
    for (size_t level = 1; level < outLevels; ++level) {
      const Grid<T> &g = outLinear.getGridForLevel(level);
      auto &dst = outFlags[level];
      for (size_t i = 0; i < g.size3; ++i) {
        size_t baseId = outLinear.getIndexOfCellOnOtherLevel(level, 0, i);
        if (inFootprint(baseId))
          dst.push_back(i);
      }
      tools::sortAndEraseDuplicate(dst);
    }
  }
}

#endif
