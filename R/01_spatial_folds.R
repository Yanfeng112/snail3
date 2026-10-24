# Whole 20 km blocks remain together; assignments are saved before model fitting.
block_ids <- function(data, cfg) {
  origin <- unlist(cfg$block_origin_m)
  if (length(origin) != 2) {
    stop("Specify the x/y origin used for spatial blocks")
  }
  paste(floor((data$x - origin[1]) / 20000),
        floor((data$y - origin[2]) / 20000), sep = "_")
}

make_folds <- function(data, cfg) {
  blocks <- block_ids(data, cfg)
  sizes <- table(blocks)
  if (length(sizes) < 10) {
    stop("Ten-fold CV requires at least ten occupied blocks")
  }
  assignments <- list()
  for (repeat_id in 1:10) {
    set.seed(42 + repeat_id - 1)
    shuffled <- sample(names(sizes), length(sizes))
    ordered <- shuffled[order(-as.numeric(sizes[shuffled]), method = "radix")]
    load <- numeric(10)
    allocation <- setNames(integer(length(sizes)), names(sizes))
    for (block in ordered) {
      available <- which(load == min(load))
      fold <- available[sample.int(length(available), 1)]
      allocation[block] <- fold
      load[fold] <- load[fold] + sizes[block]
    }
    assignments[[repeat_id]] <- data.frame(
      survey_id = data$survey_id, block_id = blocks, repeat_id = repeat_id,
      fold = unname(allocation[blocks])
    )
  }
  do.call(rbind, assignments)
}

get_mesh <- function(cfg, region) {
  specification <- cfg$meshes[[region]]
  if (!is.null(specification$rds)) {
    return(readRDS(specification$rds))
  }
  needed <- c("max_edge_m", "cutoff_m", "offset_m", "boundary_file")
  if (any(vapply(specification[needed], is.null, logical(1)))) {
    stop("Supply Mesh 3 or its boundary and mesh parameters: ", region)
  }
  boundary <- sf::st_read(specification$boundary_file, quiet = TRUE)
  boundary <- sf::st_transform(boundary, cfg$xy_crs)
  boundary <- sf::st_union(sf::st_geometry(boundary))
  segments <- INLA::inla.sp2segment(as(boundary, "Spatial"))
  segments$loc <- segments$loc / cfg$spde_coordinate_divisor
  INLA::inla.mesh.2d(
    boundary = segments,
    max.edge = unlist(specification$max_edge_m) / cfg$spde_coordinate_divisor,
    cutoff = specification$cutoff_m / cfg$spde_coordinate_divisor,
    offset = unlist(specification$offset_m) / cfg$spde_coordinate_divisor
  )
}
